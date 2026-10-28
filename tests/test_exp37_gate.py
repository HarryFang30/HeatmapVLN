"""EXP-37's offline gate: ``hm3d_route_gate.py decode --on-mode base-stop`` and ``scripts/exp37/gate_verdict.py``.

The decode is exercised with a stub model and processor (no weights): the base decides STOP from its first
generated token, the adapter decodes with STOP masked at that position only, every row names its mode, and
``score`` refuses mixed modes.  The verdict is read from synthetic ``hm3d.json`` / ``eval_pairs.py`` outputs
built with the gate's own ``summarize``.
"""

from __future__ import annotations

import contextlib
import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

from scripts.exp28 import hm3d_route_gate as g
from scripts.exp37 import gate_verdict as v
from tests.test_hm3d_route_gate import (  # noqa: F401  (flat_navmesh is a fixture)
    _decoded,
    _rows,
    _score_argv,
    _select,
    _world,
    _write_decoded,
    flat_navmesh,
)

STOP, SSTOP, LEFT, RIGHT, DOWN, PIX, END = 50669, 45537, 11, 12, 13, 14, 151645
TEXT = {STOP: "STOP", SSTOP: " STOP", LEFT: "←", RIGHT: "→", DOWN: "↓", PIX: "300 200"}
A = "a" * 64


# --------------------------------------------------------------------------- #
# Stubs: a tokenizer, a processor and a "model" whose answers are scripted
# --------------------------------------------------------------------------- #
class _Tokenizer:
    def __init__(self, split_stop=False, same=False):
        self.split_stop, self.same = split_stop, same

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text == "STOP":
            return [70, 71] if self.split_stop else [STOP]
        if text == " STOP":
            return [STOP] if self.same else [SSTOP]
        raise KeyError(text)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(TEXT[int(i)] for i in ids if int(i) != END)


class _Processor:
    """The prompt length grows with the conversation (turn 2 is longer); ``current`` is the prompt's text."""

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.current = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, return_tensors):
        import torch

        self.current = messages[0]["content"][0]["text"]
        n = 5 + 3 * len(messages)
        return {"input_ids": torch.ones(1, n, dtype=torch.long), "attention_mask": torch.ones(1, n, dtype=torch.long)}


class _Model:
    """``script(adapter_on, turn, prompt)`` -> (planned ids, ids when the first is suppressed)."""

    def __init__(self, processor, script):
        self.processor, self.script = processor, script
        self.adapter_on = True
        self.calls = []

    def eval(self):
        return self

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_on = False
        try:
            yield
        finally:
            self.adapter_on = True

    def generate(self, input_ids, attention_mask, **kwargs):
        import torch

        turn = 1 if input_ids.shape[1] == 8 else 2
        planned, alternative = self.script(self.adapter_on, turn, self.processor.current)
        if planned[0] in (kwargs.get("begin_suppress_tokens") or ()):
            planned = alternative
        self.calls.append({"on": self.adapter_on, "turn": turn, "prompt": self.processor.current, "kwargs": kwargs})
        new = torch.tensor([list(planned) + [END]], dtype=torch.long)
        return SimpleNamespace(sequences=torch.cat([input_ids, new], dim=1))


def _messages(text="go"):
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def _fake_lookdown_turn(messages, images, first, lookdown):
    return messages + [{"role": "assistant", "content": [{"type": "text", "text": first}]},
                       {"role": "user", "content": [{"type": "text", "text": "you can see"}]}], list(images)


@pytest.fixture()
def served(monkeypatch):
    """train_system2_lora's decode helpers, with the lookdown turn appended without images."""
    pytest.importorskip("torch")
    tsl = pytest.importorskip("scripts.train_system2_lora")
    from src.models.heatmap import native_internnav_exact as nie

    monkeypatch.setattr(tsl, "append_native_lookdown_turn", _fake_lookdown_turn)
    monkeypatch.setattr(nie, "append_native_lookdown_turn", _fake_lookdown_turn)
    return tsl


# --------------------------------------------------------------------------- #
# STOP ids and the base-stop arms
# --------------------------------------------------------------------------- #
def test_stop_ids_are_single_distinct_tokens():
    assert g.stop_token_ids(_Tokenizer()) == (STOP, SSTOP)
    with pytest.raises(ValueError, match="not a single token"):
        g.stop_token_ids(_Tokenizer(split_stop=True))
    with pytest.raises(ValueError, match="share a token id"):
        g.stop_token_ids(_Tokenizer(same=True))


def _decide(base, adapter):
    calls = []

    def decide(adapter_on, suppress):
        calls.append((adapter_on, tuple(suppress)))
        return adapter if adapter_on else base

    return decide, calls


@pytest.mark.parametrize("first", [STOP, SSTOP])
def test_a_base_stop_is_the_answer_and_the_adapter_does_not_run(first):
    decide, calls = _decide((["STOP"], first), (["→"], RIGHT))
    on, off = g.base_stop_arms(decide, (STOP, SSTOP))
    assert calls == [(False, ())]
    assert on == {"turns": ["STOP"], "class": "stop", "first_token_id": None, "stop_source": "base"}
    assert off == {"turns": ["STOP"], "class": "stop", "first_token_id": first}


def test_otherwise_the_adapter_decodes_with_stop_masked_at_its_first_position():
    decide, calls = _decide((["←"], LEFT), (["↓", "300 200"], DOWN))
    on, off = g.base_stop_arms(decide, [STOP, SSTOP])
    assert calls == [(False, ()), (True, (STOP, SSTOP))]
    assert on == {"turns": ["↓", "300 200"], "class": "pixel", "first_token_id": DOWN, "stop_source": "none"}
    assert off["class"] == "turn_left" and "stop_source" not in off
    # the adapter still ends in the STOP class (no action token: the server's fallback stop): "adapter"
    decide, _ = _decide((["←"], LEFT), ([""], None))
    on, _ = g.base_stop_arms(decide, [STOP, SSTOP])
    assert on["class"] == "stop" and on["stop_source"] == "adapter"


# --------------------------------------------------------------------------- #
# The decode itself, with the served generate calls
# --------------------------------------------------------------------------- #
SCRIPTS = {  # (base answer, adapter answer): (planned ids, ids if the first is suppressed)
    "turn": (([LEFT], [LEFT]), ([RIGHT], [RIGHT])),
    "lookdown": (([DOWN], [DOWN]), ([DOWN], [DOWN])),
    "stop": (([STOP], [LEFT]), ([STOP], [RIGHT])),
}


def _scripted(case):
    base, adapter = SCRIPTS[case]

    def script(adapter_on, turn, prompt):
        if turn == 2:
            return [PIX], [PIX]
        return adapter if adapter_on else base

    return script


@pytest.mark.parametrize("case", sorted(SCRIPTS))
def test_unmasked_it_is_greedy_native_decision_call_for_call(served, case):
    import torch

    processor = _Processor()
    model = _Model(processor, _scripted(case))
    for adapter_on in (True, False):
        with contextlib.nullcontext() if adapter_on else model.disable_adapter():
            served_turns = served.greedy_native_decision(model, processor, _messages(), [], None, torch.device("cpu"))
            n_served = len(model.calls)
            turns, first = g.greedy_first_token_decision(model, processor, _messages(), [], None, torch.device("cpu"))
        assert turns == served_turns
        assert [c["kwargs"] for c in model.calls[n_served:]] == [c["kwargs"] for c in model.calls[:n_served]]
        assert all(c["kwargs"] == served.NATIVE_GENERATE_KWARGS for c in model.calls)
        assert first == {"turn": RIGHT if adapter_on else LEFT, "lookdown": DOWN, "stop": STOP}[case]
        model.calls.clear()


def test_the_mask_reaches_turn_one_only(served):
    import torch

    processor = _Processor()
    model = _Model(processor, lambda on, turn, prompt: ([PIX], [PIX]) if turn == 2 else ([STOP], [DOWN]))
    turns, first = g.greedy_first_token_decision(model, processor, _messages(), [], None, torch.device("cpu"),
                                                 suppress_first=(STOP, SSTOP))
    assert turns == ("↓", "300 200") and first == DOWN
    assert model.calls[0]["kwargs"] == {**served.NATIVE_GENERATE_KWARGS, "begin_suppress_tokens": [STOP, SSTOP]}
    assert model.calls[1]["kwargs"] == served.NATIVE_GENERATE_KWARGS


# --------------------------------------------------------------------------- #
# decode end to end (stub model), then score
# --------------------------------------------------------------------------- #
def _flat_script(adapter_on, turn, prompt):
    """Prompts read '<episode>|<call>|<instruction>'.  The base stops at episode 1's call 0 and turns left
    elsewhere; the adapter wants STOP at every call 1 (masked: it turns right) and looks down elsewhere."""
    episode, index, _ = prompt.split("|", 2)
    if turn == 2:
        return [PIX], [PIX]
    if not adapter_on:
        return ([STOP], [LEFT]) if (episode, index) == ("1", "0") else ([LEFT], [LEFT])
    return ([STOP], [RIGHT]) if index == "1" else ([DOWN], [DOWN])


@pytest.fixture()
def stub_decode(served, monkeypatch, tmp_path):
    import torch

    from src.data import s2_sft_dataset as ds

    processor = _Processor()
    model = _Model(processor, _flat_script)

    def inference_messages(record_like, call, default_vlm_image_size=None):
        text = f"{record_like.episode_id}|{call['call_index']}|{record_like.episode['instruction']}"
        return _messages(text), [], None

    monkeypatch.setattr(ds, "inference_messages", inference_messages)
    monkeypatch.setattr(served, "init_runtime", lambda kind: SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(served, "seed_everything", lambda seed, device: None)
    monkeypatch.setattr(served, "load_system2", lambda path, device: (model, processor))
    monkeypatch.setattr(served, "load_adapter", lambda m, path: m)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x37")
    return model, adapter


def _decode_argv(records, labels, output, adapter, *extra):
    return ["decode", "--record-root", str(records), "--label-root", str(labels), "--scenes", "flatA",
            "--output", str(output), "--adapter", str(adapter), "--model-path", "none", "--device", "cpu",
            "--shuffle-instructions", *extra]


def test_decode_base_stop_rows_then_score(tmp_path, stub_decode, flat_navmesh):
    model, adapter = stub_decode
    records, labels = _world(tmp_path)
    output = tmp_path / "o4" / "hm3d.json"
    assert g.main(_decode_argv(records, labels, output, adapter, "--on-mode", "base-stop")) == 0
    rows = g.read_decode_rows(sorted(output.parent.glob("hm3d.decode.shard*.jsonl")))
    assert len(rows) == 9 and {r["on_mode"] for r in rows.values()} == {"base-stop"}
    assert {tuple(r["stop_token_ids"]) for r in rows.values()} == {(STOP, SSTOP)}

    stopped = rows["flatA_000001#0"]
    assert stopped["on"]["turns"] == ["STOP"] and stopped["on"]["stop_source"] == "base"
    assert stopped["off"]["first_token_id"] == STOP
    # its shuffled prompt is the base's call 0 of episode 1 too: the same mode decides it
    assert stopped["shuffled"]["on"]["stop_source"] == "base"
    masked = rows["flatA_000002#1"]
    assert masked["on"]["turns"] == ["→"] and masked["on"]["stop_source"] == "none"
    assert rows["flatA_000003#0"]["on"]["turns"] == ["↓", "300 200"]
    assert rows["flatA_000003#0"]["shuffled"]["on"]["turns"] == ["↓", "300 200"]

    adapter_calls = [c for c in model.calls if c["on"]]
    assert not [c for c in adapter_calls if c["prompt"].startswith("1|0|")]  # the base stopped: no adapter pass
    assert all(c["kwargs"].get("begin_suppress_tokens") == [STOP, SSTOP] for c in adapter_calls if c["turn"] == 1)
    assert all("begin_suppress_tokens" not in c["kwargs"] for c in model.calls if c["turn"] == 2 or not c["on"])

    # a resume in the other mode into the same output is refused, nothing is appended
    before = output.with_name("hm3d.decode.shard00of01.jsonl").read_text()
    with pytest.raises(SystemExit, match="on-mode"):
        g.main(_decode_argv(records, labels, output, adapter))
    assert output.with_name("hm3d.decode.shard00of01.jsonl").read_text() == before

    code = g.main(_score_argv(records, labels, flat_navmesh, output))
    report = json.loads(output.read_text())
    assert code in (0, 1) and report["valid"] and report["join"]["on_mode"] == "base-stop"
    sources = report["report_only"]["on_stop_source"]
    assert sources["all"] == {"base": 1, "none": 8} and sources["anchor"] == {"none": 3}
    assert report["O4"]["criteria"]["far_stop_vs_base"]["on_stops"] == 1


def test_score_refuses_rows_of_two_modes(tmp_path, flat_navmesh):
    records, labels = _world(tmp_path)
    states, _ = _select(records, labels)
    output = tmp_path / "o4" / "hm3d.json"
    rows = _decoded(states, shuffled_on=lambda s: s["recorded"])
    shard = _write_decoded(output, rows)
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 1  # read: the adapter is the base here
    report = json.loads(output.read_text())
    assert report["valid"] and report["join"]["on_mode"] == "adapter"  # rows from before the option
    assert "on_stop_source" not in report["report_only"]
    rows[0]["on_mode"] = "base-stop"
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2
    assert "2 on-modes" in json.loads(output.read_text())["invalid"]
    for row in rows:
        row["on_mode"] = "sampled"
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2


def test_served_adapter_reads_the_sha_before_a_mode_suffix():
    version = {"server": {"model_version": "run_x+system2-lora:" + A + "+base-stop-v1"}}
    assert g.served_adapter(version) == A
    assert g.served_adapter({"server": {"model_version": "run_x+system2-lora:" + A}}) == A
    assert g.served_adapter({"server": {"model_version": "run_x"}}) is None


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #
def _hrg(*, anchors=3529, kept=3529, on_mode="base-stop", valid=True, subset=0, adapter=A):
    rows = _rows(corrections=20, on_agree=6, off_agree=1, anchors=anchors, kept=kept, shuffled=(2, 1))
    summary = g.summarize(rows, shuffle_requested=True)
    report = {"schema": g.SCHEMA, "valid": valid, **summary,
              "join": {"adapter_sha256": adapter, "on_mode": on_mode, "stats": {"rows": len(rows)}},
              "selection": {"settings": {"max_episodes": subset}}}
    if not valid:
        report["invalid"] = "GateInvalid: something"
    return report, rows


def _pairs(correct=1475, n=1475, lower=0.55, adapter=A, **extra):
    accuracy = correct / n
    out = {"schema": v.PAIR_SCHEMA, "valid": True, "pairs": n, "episodes": 300,
           "reward_accuracy": round(accuracy, 6), "reward_accuracy_ci95": [lower, min(1.0, accuracy + 0.05)],
           "policy_accuracy": 0.4, "reference_accuracy": 0.1, "adapter_sha256": adapter,
           "g1": {"accuracy_line": 0.6, "lower_line": 0.5, "pass": accuracy >= 0.6 and lower > 0.5}}
    out.update(extra)
    return out


def _verdict(*, hrg=None, pairs=None, attempt=1, train=None, train_accuracy=None, adapter=None):
    report, rows = hrg or _hrg()
    return v.verdict(hrg=report, rows=rows, pairs_summary=pairs or _pairs(), attempt=attempt,
                     train_summary=train, train_accuracy=train_accuracy, adapter_sha256=adapter)


def test_both_gates_on_their_lines_pass():
    # 885 / 1475 is exactly 0.60; 3,247 / 3,529 is the first count >= 0.92 (3,246 is not)
    result = _verdict(hrg=_hrg(kept=3247), pairs=_pairs(correct=885, lower=0.5001), adapter=A)
    assert result["groups"] == {"G1": True, "G2": True} and result["next"] == "val" and result["all_pass"]
    assert result["G1"]["correct"] == 885 and result["G2"]["kept"] == 3247
    assert not [n for n in result["notes"] if "ledger" in n]  # the ledger's sizes
    fail = _verdict(hrg=_hrg(kept=3246), pairs=_pairs(correct=885, lower=0.5001))
    assert fail["groups"] == {"G1": True, "G2": False} and fail["next"] == "no_val" and fail["measured"]


def test_g1_needs_both_its_lines():
    lower_on_line = _verdict(pairs=_pairs(correct=1000, lower=0.50), train_accuracy=0.9)
    assert lower_on_line["G1"]["pass_accuracy"] and not lower_on_line["G1"]["pass_lower_bound"]
    under = _verdict(pairs=_pairs(correct=884, lower=0.55), train_accuracy=0.9)
    assert not under["G1"]["pass_accuracy"] and under["G1"]["pass_lower_bound"]
    assert lower_on_line["next"] == under["next"] == "no_val"


@pytest.mark.parametrize("attempt,g2_kept,train,step", [
    (1, 3529, 0.69, "retry_lr_5e-5"),  # G1 alone, underfit
    (1, 3529, 0.70, "no_val"),         # G1 alone, fitted: learnt, does not transfer
    (1, 3000, 0.50, "retry_lr_5e-5"),  # G1 and G2, underfit: the ledger's retry reads G1 only (G2 re-gated)
    (1, 3000, 0.70, "no_val"),         # G1 and G2, fitted
    (1, 3000, None, "invalid"),        # G1 and G2 and no training accuracy: the retry cannot be read
    (2, 3529, 0.50, "no_val"),         # the one retry failed
    (1, 3529, None, "invalid"),        # G1 alone and no training accuracy: the retry type cannot be read
])
def test_the_retry_rule(attempt, g2_kept, train, step):
    result = _verdict(hrg=_hrg(kept=g2_kept), pairs=_pairs(correct=800), attempt=attempt, train_accuracy=train)
    assert result["next"] == step and not result["all_pass"]
    assert result["measured"] == (step != "invalid")
    assert result["train_reward_accuracy_pairs"] is None  # given by hand


def test_g2_alone_failing_is_no_val_whatever_the_training_accuracy():
    for train in (0.5, None):
        result = _verdict(hrg=_hrg(kept=3000), pairs=_pairs(correct=885, lower=0.5001), train_accuracy=train)
        assert result["groups"] == {"G1": True, "G2": False} and result["next"] == "no_val" and result["measured"]


def test_the_training_accuracy_may_come_from_eval_pairs():
    result = _verdict(pairs=_pairs(correct=800), train=_pairs(correct=1000, n=1500))
    assert result["next"] == "retry_lr_5e-5" and result["train_reward_accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert result["train_reward_accuracy_pairs"] == 1500
    other = _verdict(pairs=_pairs(correct=800), train=_pairs(correct=1000, n=1500, adapter="b" * 64))
    assert other["next"] == "invalid" and "different adapters" in other["notes"][0]


@pytest.mark.parametrize("hrg,pairs,adapter,why", [
    (_hrg(on_mode="adapter"), None, None, "on-mode 'adapter'"),
    (_hrg(valid=False), None, None, "not valid"),
    (_hrg(subset=40), None, None, "smoke run"),
    (_hrg(adapter="b" * 64), None, None, "different adapters"),
    (None, None, "c" * 64, "different adapters"),
    (None, _pairs(schema="other"), None, "not eval_pairs.py"),
    (None, _pairs(valid=False), None, "not valid"),
    (None, _pairs(reward_accuracy=0.60001), None, "no count over"),
    (None, _pairs(correct=885, lower=0.5001, g1={"pass": False}), None, "eval_pairs.py read pass=False"),
])
def test_what_cannot_be_read_is_invalid_not_a_fail(hrg, pairs, adapter, why):
    result = _verdict(hrg=hrg, pairs=pairs, adapter=adapter, train_accuracy=0.5)
    assert result["next"] == "invalid" and not result["measured"] and not result["all_pass"]
    assert any(why in note for note in result["notes"]), result["notes"]


def test_report_only_items_are_read_and_do_not_gate():
    result = _verdict()
    extra = result["report_only"]
    assert extra["O4_correction"]["value"] == 0.25 and extra["O4_correction"]["n"] == 20
    assert extra["O5"]["ratio"] == pytest.approx(0.2) and extra["O5"]["delta_corr"] == 0.25
    assert extra["L"]["n"] == 20 and "exp32_line_pass" in extra["L"]
    assert result["G1"]["policy_accuracy"] == 0.4 and result["G1"]["reference_accuracy"] == 0.1
    assert result["next"] == "val"
    assert Fraction(result["G2"]["kept"], result["G2"]["n"]) == 1


def test_cli_writes_the_verdict_and_exits_like_the_stop_gate(tmp_path, capsys):
    report, rows = _hrg()
    hrg = tmp_path / "o4" / "hm3d.json"
    hrg.parent.mkdir()
    hrg.write_text(json.dumps(report))
    hrg.with_suffix(".calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pairs = tmp_path / "g1.json"
    out = tmp_path / "verdict.json"

    def run(*extra):
        return v.main(["--hrg", str(hrg), "--pairs", str(pairs), "--output", str(out), *extra])

    pairs.write_text(json.dumps(_pairs()))
    assert run() == 0 and json.loads(out.read_text())["next"] == "val"
    assert "PASS EXP-37 offline gate attempt 1: G1=pass G2=pass -> next: val" in capsys.readouterr().out
    pairs.write_text(json.dumps(_pairs(correct=800)))
    assert run("--train-reward-accuracy", "0.6") == 1 and json.loads(out.read_text())["next"] == "retry_lr_5e-5"
    assert run("--train-reward-accuracy", "0.6", "--attempt", "2") == 1
    assert json.loads(out.read_text())["next"] == "no_val"
    assert run() == 2 and json.loads(out.read_text())["next"] == "invalid"
    pairs.write_text("{not json")
    assert run() == 2
    with pytest.raises(SystemExit):
        run("--train-reward-accuracy", "0.6", "--train-pairs", str(pairs))
