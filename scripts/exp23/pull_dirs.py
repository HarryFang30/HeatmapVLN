"""Pull directories from the Ascend box's serve_dirs.py through the ssh tunnels.

usage: pull_dirs.py DEST PORTS NAME [NAME ...]
Each worker owns one tunnel port (one ssh connection each, so they run in parallel).
A directory counts as pulled only if the extracted file count and byte total match the
server's header; then DEST/<name>/.pulled is written. Existing .pulled dirs are skipped.
"""
import json, os, queue, shutil, socket, subprocess, sys, threading, time
dest, ports, names = sys.argv[1], [int(p) for p in sys.argv[2].split(",")], sys.argv[3:]
os.makedirs(dest, exist_ok=True)
q = queue.Queue()
for n in names:
    if not os.path.exists(os.path.join(dest, n, ".pulled")):
        q.put(n)
lock = threading.Lock()
def log(*a):
    with lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)
def tally(path):
    n = b = 0
    for d, _, fs in os.walk(path):
        for f in fs:
            if f == ".pulled":
                continue
            n += 1; b += os.path.getsize(os.path.join(d, f))
    return n, b
def pull(port, name):
    s = socket.create_connection(("127.0.0.1", port), timeout=120)
    s.sendall((name + "\n").encode())
    f = s.makefile("rb")
    header = json.loads(f.readline())
    if "error" in header:
        raise RuntimeError(header)
    tmp = os.path.join(dest, f".{name}.part")
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    t0 = time.time()
    p = subprocess.Popen(["tar", "-xf", "-", "-C", tmp], stdin=subprocess.PIPE)
    for chunk in iter(lambda: f.read(1 << 20), b""):
        p.stdin.write(chunk)
    p.stdin.close(); rc = p.wait(); s.close()
    got = tally(os.path.join(tmp, name))
    if rc != 0 or got != (header["files"], header["bytes"]):
        raise RuntimeError(f"{name}: tar rc={rc} got={got} want={header}")
    shutil.rmtree(os.path.join(dest, name), ignore_errors=True)
    os.rename(os.path.join(tmp, name), os.path.join(dest, name)); os.rmdir(tmp)
    open(os.path.join(dest, name, ".pulled"), "w").write(json.dumps(header) + "\n")
    dt = time.time() - t0
    log(f"OK {name} port={port} {header['bytes']/1e6:.0f} MB {dt:.0f}s {header['bytes']/1e6/max(dt,1e-3):.1f} MB/s")
def worker(port):
    while True:
        try:
            name = q.get_nowait()
        except queue.Empty:
            return
        for attempt in range(3):
            try:
                pull(port, name); break
            except Exception as e:
                log(f"RETRY {name} port={port} attempt={attempt}: {e}"); time.sleep(10)
        else:
            log(f"FAIL {name}")
ts = [threading.Thread(target=worker, args=(p,)) for p in ports]
[t.start() for t in ts]; [t.join() for t in ts]
log("ALL_DONE")
