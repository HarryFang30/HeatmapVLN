"""The servers must accept the keepalive pings their own client sends.

Measured twice on the Ascend 910B: a canary died after one episode with "AMB3R VO
RPC returned no response for ingest_frame", and the client log showed the VO server
had sent GOAWAY with ENHANCE_YOUR_CALM / "too_many_pings".  The cause is a policy
mismatch, not a device fault and not the AICPU timeout the open-problems doc first
blamed:

  - vla_rpc's sync client opens its channel with grpc.keepalive_time_ms = 30000, so
    it pings every 30 s;
  - a gRPC server defaults to grpc.http2.min_ping_interval_without_data_ms = 300000
    and max_pings_without_data = 2, i.e. at most two pings per five minutes on a
    channel that is carrying no data, and then it closes the connection;
  - on the 910B a plan call takes about 11 s and the VO channel is idle while the
    model server works, so the pings arrive with no data on that channel and trip
    the limit.  The client has no RPC retry, so one GOAWAY takes the whole shard.

It never fired on the certified 4090 runs, which is why it reached the Ascend
bring-up undetected.  These tests read the option lists out of both servers so the
policy cannot quietly drift back to the default, and they check the numbers against
the client's own ping interval rather than against a literal.

AST only: importing either server needs torch and the model stack.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL_SERVER = REPO / "scripts" / "evaluation" / "rpc_model_server.py"
VO_SERVER = REPO / "scripts" / "amb3r_vo" / "rpc_amb3r_vo_server.py"

# vla_rpc/client/sync_client.py: ("grpc.keepalive_time_ms", 30000).  vla_rpc lives
# outside this repository (PPA_EVAL_RPC_ROOT), so the value is pinned here and the
# assertions below are written against it.
CLIENT_KEEPALIVE_TIME_MS = 30000


def _grpc_options(path: Path) -> dict[str, object]:
    """Every ("grpc.*", value) pair the file builds, by option name.

    Collected from the whole module rather than from one list, because the servers
    build their options in several steps (a base list, then the instance-only and
    keepalive additions).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    options: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
            continue
        key, value = node.elts
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        if not key.value.startswith("grpc."):
            continue
        options[key.value] = value.value if isinstance(value, ast.Constant) else value
    return options


def test_both_servers_tolerate_the_clients_ping_interval():
    for path in (MODEL_SERVER, VO_SERVER):
        options = _grpc_options(path)
        interval = options.get("grpc.http2.min_ping_interval_without_data_ms")
        assert isinstance(interval, int), f"{path.name} does not set the minimum ping interval"
        # The server must not call a ping "too frequent" that its own client sends on
        # schedule.  Equal would already be a race, so require headroom.
        assert interval < CLIENT_KEEPALIVE_TIME_MS, (
            f"{path.name} allows a ping only every {interval} ms while the client pings "
            f"every {CLIENT_KEEPALIVE_TIME_MS} ms"
        )


def test_both_servers_allow_pings_on_an_idle_channel():
    for path in (MODEL_SERVER, VO_SERVER):
        options = _grpc_options(path)
        # The VO channel is idle for minutes at a time while the model server works,
        # which is exactly when the client's keepalive pings arrive.
        assert options.get("grpc.keepalive_permit_without_calls") == 1, (
            f"{path.name} must permit keepalive pings with no call in flight"
        )
        # 0 means no cap.  The default of 2 is what sent the GOAWAY.
        assert options.get("grpc.http2.max_pings_without_data") == 0, (
            f"{path.name} must not cap the number of pings sent without data"
        )


def test_the_keepalive_policy_is_not_gated_behind_a_flag():
    """It has to hold for the certified CUDA launcher too, which passes no flags.

    The instance token and SO_REUSEPORT are deliberately opt-in; this one is not,
    because the failure it prevents is a dropped connection rather than a change in
    what the server computes.
    """
    for path in (MODEL_SERVER, VO_SERVER):
        source = path.read_text(encoding="utf-8")
        head, _, tail = source.partition('("grpc.http2.min_ping_interval_without_data_ms"')
        assert tail, path.name
        # The nearest enclosing `if` above the keepalive block must not be the
        # instance-token one; the additions sit before it in both files.
        last_if = head.rfind("\n    if ")
        last_options = head.rfind("options += [")
        assert last_options > last_if, (
            f"{path.name} appears to gate the keepalive options behind a condition"
        )
