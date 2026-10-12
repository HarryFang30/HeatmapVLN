"""Loopback-only, read-only tar streamer for moving data off the 910B through the ssh tunnels.

The tunnels are -L forwards opened from the 4090 container, so the 910B side has to
listen and the container pulls (scripts/exp23/pull_dirs.py).  A client sends one line
'<name>\\n' (a direct child of ROOT matching [A-Za-z0-9_-]+); the server replies with one
JSON header line {"files": n, "bytes": total} and then an uncompressed tar of ROOT/<name>.
It listens on 127.0.0.1 only, on the given ports (tunnel ports nothing else is using).

    python serve_dirs.py ROOT PORT[,PORT...]
"""
import json
import os
import re
import socketserver
import subprocess
import sys
import threading

ROOT = os.path.realpath(sys.argv[1])
PORTS = [int(p) for p in sys.argv[2].split(",")]
NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        name = self.rfile.readline(256).decode().strip()
        path = os.path.join(ROOT, name)
        if not NAME.match(name) or not os.path.isdir(path):
            self.wfile.write(b'{"error": "bad name"}\n')
            return
        n = b = 0
        for d, _, fs in os.walk(path):
            for f in fs:
                n += 1
                b += os.path.getsize(os.path.join(d, f))
        self.wfile.write((json.dumps({"files": n, "bytes": b}) + "\n").encode())
        self.wfile.flush()
        p = subprocess.Popen(["tar", "-C", ROOT, "-cf", "-", name], stdout=subprocess.PIPE)
        for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
            self.wfile.write(chunk)
        p.wait()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


for port in PORTS:
    threading.Thread(target=Server(("127.0.0.1", port), Handler).serve_forever, daemon=True).start()
print("serving", ROOT, PORTS, flush=True)
threading.Event().wait()
