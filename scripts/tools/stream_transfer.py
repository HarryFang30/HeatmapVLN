#!/usr/bin/env python3
"""Move files from the 4090 container to the 910B through the existing ssh tunnels.

The tunnels are ``-L`` forwards opened from the container (design §4): every
connection starts on the 4090, so the 910B side listens and the 4090 side
pushes.  ``recv`` binds 127.0.0.1 only, on tunnel ports nothing else is using,
and writes inside ``--dest`` alone.  It refuses a file that would take the
directory past ``--cap-gb`` (the pusher backs off and retries), verifies every
file's sha256 before an atomic rename, and answers EXISTS for a file it already
holds, so a restarted push skips what arrived.

    # 910B (inside zhr_1)
    stream_transfer.py recv --dest ~/work/zhr/zhr_1/stream/round1 --ports 52410,52411 --cap-gb 10
    # 4090 container
    stream_transfer.py push --src /workspace/exp23/round1/stream --ports 52410,52411 --start-shard 0

Protocol, one file per connection: the client sends one JSON line
``{"name", "size", "sha256"}``; the server answers ``GO``, ``BUSY`` or
``EXISTS``; after ``GO`` exactly ``size`` bytes follow and the server answers
``DONE`` or ``BAD``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?$")
CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_bytes(root: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except FileNotFoundError:
                pass  # the trainer deleted it meanwhile
    return total


# --------------------------------------------------------------------------- #
# Receiver (910B)
# --------------------------------------------------------------------------- #
class _State:
    """One per receiver process, shared by the listeners on every port."""

    def __init__(self, dest: Path, cap_bytes: int):
        self.dest = dest
        self.cap_bytes = cap_bytes
        self.reserved = 0  # bytes of uploads in flight, counted against the cap
        self.lock = threading.Lock()


class _Receiver(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: _State = self.server.state  # type: ignore[attr-defined]
        try:
            header = json.loads(self.rfile.readline(4096))
            name, size, sha = str(header["name"]), int(header["size"]), str(header["sha256"])
        except Exception:
            self.wfile.write(b"BAD\n")
            return
        if not NAME.match(name) or ".." in name.split("/") or size < 0:
            self.wfile.write(b"BAD\n")
            return
        final = state.dest / name
        if final.is_file() and final.stat().st_size == size:
            self.wfile.write(b"EXISTS\n")
            return
        with state.lock:
            if _dir_bytes(state.dest) + state.reserved + size > state.cap_bytes:
                self.wfile.write(b"BUSY\n")
                return
            state.reserved += size
        tmp = final.with_name(f".{final.name}.{threading.get_ident()}.part")
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            self.wfile.write(b"GO\n")
            self.wfile.flush()
            digest = hashlib.sha256()
            remaining = size
            with open(tmp, "wb") as out:
                while remaining:
                    chunk = self.rfile.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            if remaining or digest.hexdigest() != sha:
                self.wfile.write(b"BAD\n")
                return
            os.replace(tmp, final)
            self.wfile.write(b"DONE\n")
            print(f"{time.strftime('%H:%M:%S')} recv {name} {size / 1e6:.1f} MB", flush=True)
        finally:
            tmp.unlink(missing_ok=True)
            with state.lock:
                state.reserved -= size


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def recv(args: argparse.Namespace) -> int:
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    state = _State(dest, int(args.cap_gb * 1e9))
    for port in (int(p) for p in args.ports.split(",")):
        server = _Server(("127.0.0.1", port), _Receiver)
        server.state = state  # type: ignore[attr-defined]
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"recv -> {dest} ports {args.ports} cap {args.cap_gb} GB", flush=True)
    threading.Event().wait()
    return 0


# --------------------------------------------------------------------------- #
# Pusher (4090 container)
# --------------------------------------------------------------------------- #
def _push_one(port: int, src: Path, name: str, timeout: float) -> str:
    path = src / name
    size = path.stat().st_size
    header = json.dumps({"name": name, "size": size, "sha256": _sha256(path)}).encode() + b"\n"
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(header)
        reply = sock.makefile("rb").readline().strip().decode()
        if reply != "GO":
            return reply or "NOREPLY"
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK), b""):
                sock.sendall(chunk)
        return sock.makefile("rb").readline().strip().decode() or "NOREPLY"


def push(args: argparse.Namespace) -> int:
    src = Path(args.src)
    if args.files:
        names = [line.strip() for line in Path(args.files).read_text().splitlines() if line.strip()]
    else:
        manifest = json.loads((src / "stream_manifest.json").read_text())
        names = ["stream_manifest.json"] + [s["name"] for s in manifest["shards"][args.start_shard:]]
    work: "queue.Queue[str]" = queue.Queue()
    for name in names:
        work.put(name)
    failed: list[str] = []
    lock = threading.Lock()

    def worker(port: int) -> None:
        while True:
            try:
                name = work.get_nowait()
            except queue.Empty:
                return
            attempts = 0
            busy_since, last_note = 0.0, 0.0
            while True:
                try:
                    reply = _push_one(port, src, name, args.timeout)
                except OSError as err:
                    reply = f"ERR {err}"
                if reply in ("DONE", "EXISTS"):
                    with lock:
                        print(f"{time.strftime('%H:%M:%S')} {reply} {name} port={port}", flush=True)
                    break
                if reply == "BUSY":
                    busy_since = busy_since or time.time()
                    if time.time() - last_note >= 60:
                        with lock:  # a stall shows up in the log instead of retrying silently
                            print(f"{time.strftime('%H:%M:%S')} BUSY {name} port={port} for {time.time() - busy_since:.0f}s", flush=True)
                        last_note = time.time()
                    time.sleep(args.busy_sleep)  # the buffer is full: the trainer has not caught up
                    continue
                attempts += 1
                if attempts >= args.retries:
                    with lock:
                        failed.append(name)
                        print(f"FAIL {name}: {reply}", flush=True)
                    break
                time.sleep(10)

    threads = [threading.Thread(target=worker, args=(int(p),)) for p in args.ports.split(",")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"push finished: {len(names) - len(failed)}/{len(names)} ok", flush=True)
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recv")
    r.add_argument("--dest", required=True)
    r.add_argument("--ports", required=True)
    r.add_argument("--cap-gb", type=float, default=10.0)
    p = sub.add_parser("push")
    p.add_argument("--src", required=True)
    p.add_argument("--ports", required=True)
    p.add_argument("--start-shard", type=int, default=0)
    p.add_argument("--files", default="", help="push these paths (relative to --src, one per line) instead of a stream")
    p.add_argument("--busy-sleep", type=float, default=15.0)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    return recv(args) if args.cmd == "recv" else push(args)


if __name__ == "__main__":
    sys.exit(main())
