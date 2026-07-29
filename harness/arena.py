#!/usr/bin/env python3
"""
vLLM Spark arena harness.

Same contract as the llama.cpp arena — paired ratios, token identity, thermal
gate, warmup discarded — over a different substrate, with one extra gate.

WHAT IS DIFFERENT HERE

  1. No build. vLLM on this fleet is a pip wheel with precompiled CUDA
     extensions and nobody has built it from source on GB10. A submission is a
     directory under patches/<name>/ containing sitecustomize.py, placed on
     PYTHONPATH for the candidate arm only. Startup injection matters: it reaches
     vLLM's spawned engine and worker processes, which a post-import monkeypatch
     would not.

  2. The serve config is PINNED, as contract rather than convenience. vLLM is
     only cross-boot deterministic under a pinned config (measured 2026-07-28,
     three boots), and gates 2 and 3 compare a baseline server against a
     candidate server — two processes. Without the pins the gate fires on boot
     noise instead of on the patch.

  3. Gate 3 checks SPEED as well as identity. In the llama.cpp arena the editable
     surface is CUDA kernels, so a submission physically cannot memoize a
     response. Here it is arbitrary Python: a patch could cache completions keyed
     on the prompt, return byte-identical tokens at absurd speed, and pass gates
     1, 2 and 4 cleanly. Freshly generated held-out prompts defeat that — a cache
     cannot hit an input nobody has seen — but only if the held-out arm is timed.
     Identity alone would let the cheat through.

Stdlib only. Runs on the node.

    python3 harness/arena.py baseline --target <t>
    python3 harness/arena.py bench    --target <t> --patch <name>
    python3 harness/arena.py heldout  --target <t> --patch <name>
"""

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads((ROOT / "benchmark.json").read_text())
VENV = Path(os.path.expanduser(CONTRACT["substrate"]["wheel"]["venv"]))


class Out:
    BOLD, DIM, RED, GRN, YEL, OFF = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"
    step = staticmethod(lambda m: print(f"{Out.BOLD}==>{Out.OFF} {m}", flush=True))
    info = staticmethod(lambda m: print(f"    {Out.DIM}{m}{Out.OFF}", flush=True))
    ok = staticmethod(lambda m: print(f"    {Out.GRN}PASS{Out.OFF} {m}", flush=True))
    fail = staticmethod(lambda m: print(f"    {Out.RED}FAIL{Out.OFF} {m}", flush=True))
    warn = staticmethod(lambda m: print(f"    {Out.YEL}warn{Out.OFF} {m}", flush=True))


def die(msg, code=1):
    print(f"\n{Out.RED}{Out.BOLD}arena: {msg}{Out.OFF}\n", file=sys.stderr)
    sys.exit(code)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------- contract

def load_target(slug):
    path = ROOT / "targets" / slug / "target.json"
    if not path.exists():
        avail = sorted(p.name for p in (ROOT / "targets").iterdir() if p.is_dir())
        die(f"no target '{slug}'. Available: {', '.join(avail) or '(none)'}")
    t = json.loads(path.read_text())
    t["_dir"], t["_slug"] = path.parent, slug

    # Hard rule: a target must point at weights a contributor can DOWNLOAD.
    # Requiring a local quantization puts hours of work in front of someone's
    # first line of code, and most people stop there.
    w = t.get("weights")
    if t.get("status") != "blocked" and (not w or not w.get("hfRepo")):
        die(f"target '{slug}' declares no downloadable weights.\n"
            f"    Add a weights block naming a PUBLIC Hugging Face repo, verified not gated\n"
            f"    and not private. See policy.publicWeights.")
    if w and not w.get("public", False):
        Out.warn(f"{slug}: weights are NOT publicly downloadable — this must not be "
                 f"anyone's entry point")
    if t.get("status") == "blocked":
        blockers = t.get("blockers", [])
        hard = [b for b in blockers if b.get("severity") == "hard"]
        die(f"target '{slug}' is blocked and cannot be measured yet.\n" +
            "".join(f"    - {b['id']}: {b['what'][:150]}...\n" for b in hard) +
            "    It is scaffolded so the work is visible, not to be run.")
    return t


def prompt_set(target):
    """Literal prompt files, fixed order. Never generated per run: the site's own
    harness cache-busts with a unique prefix, which is right for throughput and
    fatal here — token identity needs the same bytes every time."""
    d = target["_dir"] / "prompts"
    if not d.exists():
        die(f"target {target['_slug']} has no prompts/")
    out = []
    for p in sorted(d.glob("*.txt")):
        meta = target.get("promptSettings", {}).get(p.stem, {})
        out.append({"id": p.stem, "text": p.read_text(),
                    "maxTokens": meta.get("maxTokens", target.get("defaultMaxTokens", 256))})
    return out


def wheel_version():
    try:
        return subprocess.run([str(VENV / "bin" / "python3"), "-c",
                               "import vllm; print(vllm.__version__)"],
                              capture_output=True, text=True, timeout=180,
                              check=True).stdout.strip()
    except Exception as e:
        die(f"could not read vLLM version from {VENV}: {e}")


def model_fingerprint(target, full=False):
    """Shard names + sizes by default; full content hashing is opt-in because it
    costs minutes on a 20 GB checkpoint. The referee turns it on."""
    p = Path(os.path.expanduser(target["model"]))
    if not p.exists():
        die(f"model missing: {p}")
    files = sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
    if not files:
        die(f"no safetensors under {p}")
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        if full:
            with open(f, "rb") as fh:
                for blk in iter(lambda: fh.read(8 << 20), b""):
                    h.update(blk)
        else:
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()


# -------------------------------------------------------------------- thermal

def sh(cmd, host=None, timeout=120, check=False):
    """Run a shell command locally or on another node.

    ALWAYS over OpenSSH, never a Tailscale SSH session: tailscaled terminates
    the session itself and never runs PAM, so pam_limits never fires and every
    child keeps systemd's 8 MB memlock. Ray workers inherit that, and NCCL then
    silently falls back to TCP sockets even when the root memlock fix is applied
    correctly. `ssh localhost` is used for the local node for the same reason.
    """
    argv = (["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
             host or "localhost", cmd] if host is not None or True else ["bash", "-lc", cmd])
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        die(f"[{host or 'localhost'}] failed: {cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def gpu(metric, host=None):
    try:
        cmd = f"nvidia-smi --query-gpu={metric} --format=csv,noheader,nounits"
        if host is None:
            out = subprocess.run(cmd.split(), capture_output=True, text=True,
                                 timeout=15, check=True).stdout
        else:
            out = sh(cmd, host=host, timeout=30)
        return float(out.split("\n")[0])
    except Exception:
        return None


def thermal_gate(hosts=(None,)):
    """Every node starts an arm from the same thermal state.

    Relative to a per-run idle floor, not an absolute ceiling. Measured
    2026-07-29: the same node, verifiably idle, sat at 52 C while nvidia-smi
    also falsely reported 96% utilisation -- a 46 C ceiling taken from
    yesterday's 43 C idle then waits forever. The requirement was never an
    absolute number, only that the arms are comparable.

    On a cluster this runs on EVERY node. One hot worker makes a paired ratio
    meaningless, and it is the node nobody is watching.
    """
    cfg = CONTRACT["measurement"]["thermalGate"]
    budget = cfg["maxWaitSeconds"]
    tol = cfg.get("toleranceC", 3)
    temps = {}
    for h in hosts:
        label = h or "head"
        t = gpu("temperature.gpu", host=h)
        if t is None:
            Out.warn(f"{label}: no nvidia-smi temperature — thermal gate skipped, advisory")
            continue
        floor = min([gpu("temperature.gpu", host=h) or t for _ in range(3)] + [t])
        ceiling = floor + tol
        waited = 0
        while t > ceiling and waited < budget:
            Out.info(f"thermal gate [{label}]: {t:.0f} C > {ceiling:.0f} C "
                     f"(floor {floor:.0f}), waiting ({waited}s/{budget}s)")
            time.sleep(20)
            waited += 20
            t = gpu("temperature.gpu", host=h)
        if t > ceiling:
            die(f"thermal gate [{label}] never cleared: {t:.0f} C > {ceiling:.0f} C after {budget}s")
        Out.info(f"thermal gate [{label}]: {t:.0f} C (floor {floor:.0f} + {tol})")
        temps[label] = t
    return temps.get("head")


# -------------------------------------------------------------------- cluster

class Cluster:
    """A Ray cluster across N Sparks, for targets that declare `cluster`.

    Every precondition is asserted BEFORE the model loads. On this target that
    is 222 GB and about four minutes, and each of the failures below produced a
    misleading symptom that pointed somewhere else:

      memlock 8 MB   -> NCCL silently uses TCP sockets; it still serves, and the
                        paired ratio is then measured on the wrong interconnect.
      hostname->lo   -> Gloo binds 127.0.0.1 and the peer gets 'connection
                        refused', which reads as an interconnect fault.
      nvcc off PATH  -> vLLM reports the DSv4 decode kernel missing, naming a
                        component that is present and working.
      stale raylet   -> `ray start` ATTACHES to the survivor and inherits its
                        environment, manufacturing bugs that are not there.
    """

    def __init__(self, target, env_extra):
        c = target["cluster"]
        self.cfg = c
        self.nic = c.get("nic")
        self.head_ip = c["head"]["ip"]
        self.workers = c.get("workers", [])
        self.hosts = [None] + [w["host"] for w in self.workers]   # None == head
        self.ray = str(VENV / "bin" / "ray")
        self.env_extra = env_extra
        self.expect_gpus = c.get("nodes", 1 + len(self.workers))

    def _env_str(self, host_ip):
        env = dict(self.env_extra)
        env["VLLM_HOST_IP"] = host_ip
        if self.nic:
            env.setdefault("NCCL_SOCKET_IFNAME", self.nic)
            env.setdefault("GLOO_SOCKET_IFNAME", self.nic)
        return " ".join(f"{k}={v}" for k, v in env.items())

    def _ips(self):
        return [self.head_ip] + [w["ip"] for w in self.workers]

    def teardown(self):
        for h in self.hosts:
            # [r]aylet character class: an unguarded pattern matches the ssh
            # command line running it and kills the session mid-teardown.
            sh(f"{self.ray} stop --force >/dev/null 2>&1; "
               f"pkill -9 -f '[r]aylet --' 2>/dev/null; "
               f"pkill -9 -f '[g]cs_server' 2>/dev/null; true", host=h, timeout=180)
        time.sleep(5)
        for h in self.hosts:
            # pgrep -c prints 0 AND exits non-zero, so `|| echo 0` gives "0\n0".
            n = sh("pgrep -cf '[r]aylet --'", host=h) or "0"
            if n.split("\n")[0].strip() not in ("0", ""):
                die(f"[{h or 'head'}] {n} raylet(s) survived teardown. `ray start` would "
                    f"ATTACH to the survivor and inherit its environment.")

    def preflight(self):
        for h, ip in zip(self.hosts, self._ips()):
            label = h or "head"
            ml = sh("ulimit -l", host=h)
            if ml != "unlimited":
                die(f"[{label}] memlock is {ml}, not unlimited. NCCL cannot use RDMA and "
                    f"will fall back to TCP sockets without saying so. Fix limits.d, and "
                    f"check you are not reading this through a Tailscale SSH session — "
                    f"tailscaled runs no PAM, so it always reports the 8 MB default.")
            Out.info(f"[{label}] memlock unlimited, ip {ip}")
        for probe in self.cfg.get("preflight", []):
            for h in self.hosts:
                label = h or "head"
                got = sh(f"env {self._env_str(self.head_ip)} {VENV}/bin/python -c "
                         f"\"{probe['python']}\"", host=h, timeout=300)
                got = (got.split("\n")[-1] if got else "").strip()
                if got != str(probe["expect"]):
                    die(f"[{label}] preflight '{probe['name']}' returned {got!r}, "
                        f"expected {probe['expect']!r}.\n    {probe.get('hint', '')}")
                Out.info(f"[{label}] preflight {probe['name']} = {got}")

    def start(self):
        self.teardown()
        self.preflight()
        Out.info(f"ray head on {self.head_ip}")
        sh(f"env {self._env_str(self.head_ip)} {self.ray} start --head "
           f"--node-ip-address={self.head_ip} --port=6379 --num-gpus=1 --disable-usage-stats",
           host=None, timeout=300, check=True)
        time.sleep(5)
        for w in self.workers:
            Out.info(f"ray worker {w['ip']} via {w['host']}")
            sh(f"env {self._env_str(w['ip'])} {self.ray} start --address={self.head_ip}:6379 "
               f"--node-ip-address={w['ip']} --num-gpus=1",
               host=w["host"], timeout=300, check=True)
        time.sleep(8)
        got = sh(f"env {self._env_str(self.head_ip)} {VENV}/bin/python -c "
                 f"\"import ray; ray.init(address='{self.head_ip}:6379'); "
                 f"print(int(ray.cluster_resources().get('GPU',0)))\"", timeout=300)
        got = (got.split("\n")[-1] if got else "").strip()
        if got != str(self.expect_gpus):
            die(f"ray sees {got} GPU(s), expected {self.expect_gpus}. A cluster that came "
                f"up short produces a plausible number from the wrong hardware, which is "
                f"worse than an error.")
        Out.ok(f"cluster up: {self.expect_gpus} GPUs across {len(self.hosts)} node(s)")

    def assert_transport(self, log_path):
        """RDMA or sockets? The fallback is silent and it still serves."""
        if not self.cfg.get("requireRdma", True):
            return
        text = Path(log_path).read_text(errors="ignore")
        if "NET/IB" in text:
            dev = [l for l in text.splitlines() if "Made virtual device" in l]
            Out.ok("NCCL transport = IB/RoCE" + (f"  ({dev[0].split('NET/IB')[-1].strip()[:60]})" if dev else ""))
        elif "NET/Socket" in text:
            die("NCCL fell back to TCP SOCKETS. It serves, but this is not the recipe's "
                "interconnect and a paired ratio measured over it is not this hardware.")
        else:
            Out.warn("could not determine NCCL transport from the serve log "
                     "(set NCCL_DEBUG=INFO in the target env to make this checkable)")


# --------------------------------------------------------------------- server

class Server:
    def __init__(self, target, patch, log_path):
        self.target, self.patch, self.log_path = target, patch, log_path
        self.port = target.get("port", 8000)
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = None

    def _env(self):
        env = dict(os.environ)
        env["PATH"] = f"{VENV / 'bin'}:/usr/local/cuda/bin:" + env.get("PATH", "")
        # Shared across boots on purpose — a per-boot compile cache is one of the
        # things that stops vLLM being deterministic across restarts.
        env["VLLM_CACHE_ROOT"] = os.path.expanduser(
            self.target.get("cacheRoot", "~/.cache/vllm-arena"))
        for k, v in (self.target.get("env") or {}).items():
            env[k] = os.path.expanduser(str(v))
        # patchRequired is part of the BASELINE, not a submission: this target
        # does not serve at all without it, so both arms carry it and it can
        # never be someone's win. The candidate patch stacks on top.
        for name in [self.target.get("patchRequired"), self.patch]:
            if not name:
                continue
            d = ROOT / "patches" / name
            if not (d / "sitecustomize.py").exists():
                die(f"patch '{name}' has no sitecustomize.py at {d}")
            # PYTHONPATH, not an import hook: this must apply at interpreter
            # startup so it reaches the engine and worker processes vLLM spawns.
            env["PYTHONPATH"] = str(d) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    def _argv(self):
        argv = [str(VENV / "bin" / "vllm"), "serve", os.path.expanduser(self.target["model"])]
        for a in self.target["serveArgs"]:
            argv.append(os.path.expanduser(str(a)))
        return argv + ["--host", "127.0.0.1", "--port", str(self.port)]

    def __enter__(self):
        argv = self._argv()
        Out.info("serve: " + " ".join(argv)
                 + (f"   [patch {self.patch}]" if self.patch else "   [baseline]"))
        env = self._env()
        # A clustered target brings its Ray cluster up per arm, not once per
        # session: an arm must not inherit the previous arm's raylets, which is
        # how a candidate tree silently gets measured against itself.
        self.cluster = None
        if self.target.get("cluster"):
            passthrough = {k: env[k] for k in
                           ("PATH", "VLLM_CACHE_ROOT", "PYTHONPATH", "NCCL_DEBUG",
                            "NCCL_DEBUG_SUBSYS", "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
                            "RAY_local_fs_capacity_threshold", "RAY_memory_monitor_refresh_ms")
                           if k in env}
            self.cluster = Cluster(self.target, passthrough)
            self.cluster.start()
            env["VLLM_HOST_IP"] = self.cluster.head_ip
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        deadline = time.time() + self.target.get("startupTimeout", 1800)
        while time.time() < deadline:
            if self.proc.poll() is not None:
                die(f"vllm serve exited rc={self.proc.returncode}:\n"
                    f"{self.log_path.read_text()[-2500:]}")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(3)
        else:
            self.__exit__(None, None, None)
            die("vllm serve never became healthy")
        if self.cluster:
            self.cluster.assert_transport(self.log_path)
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            # by pid, never pkill -f: over ssh your own command line contains the
            # pattern and you kill the session.
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        try:
            self.log.close()
        except Exception:
            pass
        if getattr(self, "cluster", None):
            self.cluster.teardown()
        return False

    def complete(self, prompt, max_tokens):
        body = {
            "model": self.target.get("servedModelName", "arena"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "top_k": 1, "top_p": 1.0,
            "seed": self.target.get("seed", 0), "stream": True,
            "stream_options": {"include_usage": True},
            **(self.target.get("extraBody") or {}),
        }
        req = urllib.request.Request(f"{self.base}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        chunks, usage, t_first = [], {}, None
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.target.get("requestTimeout", 1800)) as res:
            for raw in res:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices", []):
                    piece = ch.get("delta", {}).get("content")
                    if piece:
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks.append(piece)
        t_end = time.perf_counter()
        if t_first is None:
            die("server returned no content. If this model does interleaved thinking the "
                "text is in reasoning_content — disable it in the target's serveArgs.")
        text = "".join(chunks)
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens") or len(chunks)
        ttft, dec = t_first - t0, t_end - t_first
        return {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "promptTokens": pt, "completionTokens": ct, "ttftSeconds": round(ttft, 4),
                "decodeTps": round((ct - 1) / dec, 3) if dec > 0 and ct > 1 else None,
                "prefillTps": round(pt / ttft, 2) if pt and ttft > 0 else None}


def run_arm(target, patch, prompts, label, tag):
    Out.step(f"arm: {label}")
    # Gate every node, not just the head. One hot worker is enough to make a
    # paired ratio meaningless, and it is the node nobody is watching.
    c = target.get("cluster")
    hosts = tuple([None] + [w["host"] for w in c.get("workers", [])]) if c else (None,)
    temp0 = thermal_gate(hosts)
    logs = ROOT / "results" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    with Server(target, patch, logs / f"{target['_slug']}-{tag}.log") as srv:
        for _ in range(CONTRACT["measurement"]["warmupRequests"]):
            srv.complete("warmup", 32)
        Out.info("discarded warmup (the first request after a boot always differs)")
        rows = []
        for p in prompts:
            r = srv.complete(p["text"], p["maxTokens"])
            r["id"] = p["id"]
            rows.append(r)
            Out.info(f"{p['id']:<18} decode {r['decodeTps']:>8} tok/s   "
                     f"prefill {str(r['prefillTps']):>9}   "
                     f"{r['promptTokens']}->{r['completionTokens']} tok   {r['sha256'][:12]}")
    return {"label": label, "patch": patch, "at": now_iso(),
            "tempStartC": temp0, "tempEndC": gpu("temperature.gpu"),
            "powerEndW": gpu("power.draw"), "prompts": rows,
            "decodeTps": statistics.median([r["decodeTps"] for r in rows if r["decodeTps"]]),
            "prefillTps": statistics.median([r["prefillTps"] for r in rows if r["prefillTps"]])}


# ------------------------------------------------------------ held-out prompts

_TOPICS = ["a ring buffer with a lock-free single-producer path",
           "a tokenizer that merges byte pairs from a frozen vocabulary",
           "a scheduler that admits requests under a fixed memory budget",
           "a cache with time-to-live eviction and hit-rate accounting",
           "a rate limiter using a sliding window over timestamps",
           "a binary format reader that validates a header before mmap"]
_SUBJECTS = ["why memory bandwidth, not FLOPs, sets decode speed on a unified-memory part",
             "how a mixture-of-experts model routes a token and what that costs in reads",
             "what a KV cache stores, and why sliding-window attention shrinks it",
             "why the first request after a server start is not comparable to the rest",
             "why greedy decoding can still be non-deterministic once a batch forms",
             "how thermal drift corrupts an unpaired throughput comparison"]


def _rand(seed_hex, *parts):
    h = hashlib.sha256(seed_hex.encode())
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode())
    return int.from_bytes(h.digest()[:8], "big")


def generate_heldout(target, seed_hex, count):
    """Determined entirely by (seed, target, index). Never stored — see cmd_heldout."""
    out = []
    for i in range(count):
        def r(*k):
            return _rand(seed_hex, target["_slug"], i, *k)
        n_filler = (0, 12, 80)[i % 3] + r("f") % 8
        if r("kind") % 2 == 0:
            task = (f"Write a complete, working Python implementation of "
                    f"{_TOPICS[r('t') % len(_TOPICS)]}. Include type hints, docstrings and "
                    f"unittest test cases. Output only code.")
        else:
            task = (f"Explain in careful technical prose: {_SUBJECTS[r('s') % len(_SUBJECTS)]}. "
                    f"Work through the arithmetic where it matters.")
        filler = "".join(f"Note {j} (ref {r('r', j) % 100000:05d}): "
                         f"{_SUBJECTS[r('fs', j) % len(_SUBJECTS)]}. " for j in range(n_filler))
        out.append({"id": f"heldout-{i:02d}",
                    "text": (f"{filler}\n\nIgnore the notes above.\n\n{task}" if filler else task),
                    "maxTokens": target.get("heldOutMaxTokens", 384)})
    return out


# ------------------------------------------------------------------- commands

def cmd_warm(args):
    """Land every prompt shape's JIT in the on-disk cache before measuring.

    Some kernels (TileLang, Triton) compile the first time they see a shape. On
    a single node that is a latency spike; under TP it is fatal, because one
    rank compiles while the other waits on the collective and the engine dies on
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS. Measured 2026-07-29: a baseline arm
    survived two prompts, hit `mhc_pre_big_fuse_with_norm_tilelang` on the
    third, and hung for the full 900 s.

    The one-request warmup in the measurement contract does not help -- it warms
    ONE shape. This warms every shape the target will actually measure, with a
    generous timeout, and throws the results away. The caches are per node and
    they drift, so both ranks compiling here is the point.

    Not a measurement: nothing it produces is recorded, and it is deliberately a
    separate command so that a run which needed warming is visible as a run that
    needed warming.
    """
    target = load_target(args.target)
    prompts = prompt_set(target)
    target = dict(target)
    target["requestTimeout"] = args.timeout
    target.setdefault("env", {})
    target["env"] = {**target["env"], "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": str(args.timeout)}
    Out.step(f"warm: {target['_slug']} — {len(prompts)} shape(s), timeout {args.timeout}s")
    logs = ROOT / "results" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    with Server(target, args.patch, logs / f"{target['_slug']}-warm.log") as srv:
        for p in prompts:
            t0 = time.perf_counter()
            try:
                # FULL maxTokens, not a truncated probe. Kernels compile per
                # shape, and the shapes a long prompt reaches at token 160 are
                # not the ones it reaches at token 16 -- a 16-token warm pass
                # returned in 1.7 s and the measured arm still ate six JIT
                # events and died. Warm exactly what will be measured.
                srv.complete(p["text"], p["maxTokens"])
                Out.ok(f"{p['id']:<16} warmed in {time.perf_counter()-t0:6.1f}s")
            except Exception as e:
                # A hang here is the JIT doing its work. Report and continue --
                # the compile still lands in the cache, which is the whole point.
                Out.warn(f"{p['id']:<16} failed after {time.perf_counter()-t0:6.1f}s "
                         f"({type(e).__name__}) — compiles may still have cached")
    Out.step("warm pass done. Re-run it; every shape should now return in seconds. "
             "If one still stalls, it is not a cold cache.")


def cmd_baseline(args):
    target = load_target(args.target)
    prompts = prompt_set(target)
    Out.step(f"baseline: {target['_slug']} ({len(prompts)} prompts, no patch)")
    Out.info(f"vLLM {wheel_version()}")
    arms = [run_arm(target, None, prompts, "baseline", f"base-{i}") for i in range(args.repeats)]

    # The FIRST boot is discarded when the compile cache was cold. Measured
    # 2026-07-28: against an empty VLLM_CACHE_ROOT, boot 1 differed from every
    # later boot on all four prompts; once warm, three consecutive boots agreed
    # on three of them. This is the request-warmup rule one level up — the first
    # boot against a cold cache is not comparable to the rest.
    if args.discard_first and len(arms) > 1:
        Out.info("discarded boot 0 (compile-cache warmup — a cold VLLM_CACHE_ROOT "
                 "changes kernel selection, so boot 0 is not comparable)")
        arms = arms[1:]

    # Prompt-level stability screening. Under the pinned config vLLM is mostly
    # cross-boot identical, but NOT reliably so per prompt: one prompt of four
    # was observed flipping between two hashes across three warm boots, while the
    # other three — including a 6,863-token one — were rock stable. So this is a
    # numerical knife-edge on a particular input, not a length effect.
    #
    # A prompt whose argmax is a coin flip cannot test anything: it would fail
    # honest submissions at random. Screen it out here, loudly, and record what
    # was dropped — silently shrinking the gate would be much worse.
    # Prompts with a previously OBSERVED flip stay excluded regardless of what
    # this run sees. Stability evidence accumulates; a single clean screen does
    # not overturn a recorded flip, and a mostly-stable prompt is the dangerous
    # kind — it fails an honest submission occasionally and reads as a
    # regression.
    known_bad = target.get("knownUnstablePrompts", {})
    stable, unstable = {}, {}
    for r in arms[0]["prompts"]:
        seen = {a_r["sha256"] for a in arms for a_r in a["prompts"] if a_r["id"] == r["id"]}
        if r["id"] in known_bad:
            unstable[r["id"]] = sorted(seen)
            Out.warn(f"{r['id']}: excluded by knownUnstablePrompts "
                     f"({known_bad[r['id']].get('note', '')[:60]}...)")
        elif len(seen) == 1:
            stable[r["id"]] = {"sha256": r["sha256"],
                               "completionTokens": r["completionTokens"],
                               "promptTokens": r["promptTokens"]}
        else:
            unstable[r["id"]] = sorted(seen)
            Out.warn(f"{r['id']}: {len(seen)} distinct outputs across {len(arms)} boots "
                     f"— excluded from the gate")
    if not stable:
        die("no prompt was stable across boots. Either the serve config is not pinned "
            "(check --max-num-seqs 1, --kv-cache-memory-bytes, --max-model-len, "
            "--gpu-memory-utilization, shared VLLM_CACHE_ROOT) or this model/config is "
            "not a viable arena target — the token-identity gate needs a stable baseline "
            "before it can judge anything.")
    if len(stable) < 2:
        die(f"only {len(stable)} stable prompt survived screening. A one-prompt gate is "
            f"too weak to catch a kernel that changed behaviour on a different shape.")
    Out.ok(f"cross-boot stability: {len(stable)}/{len(stable) + len(unstable)} prompts "
           f"identical across {len(arms)} boots")
    goldens = stable

    (target["_dir"] / "goldens.json").write_text(json.dumps(
        {"contractVersion": CONTRACT["contractVersion"], "wheel": wheel_version(),
         "recordedAt": now_iso(), "boots": len(arms), "prompts": goldens,
         "excluded": unstable,
         "excludedNote": "Prompts whose greedy output was not identical across every "
                         "measured boot. They are recorded rather than deleted: an "
                         "excluded prompt is evidence about this model's numerical "
                         "stability, and the gate must not silently shrink."}, indent=2) + "\n")
    (target["_dir"] / "baseline.json").write_text(json.dumps(
        {"contractVersion": CONTRACT["contractVersion"], "wheel": wheel_version(),
         "recordedAt": now_iso(), "node": os.uname().nodename,
         "modelFingerprint": model_fingerprint(target),
         "decodeTps": statistics.median([a["decodeTps"] for a in arms]),
         "prefillTps": statistics.median([a["prefillTps"] for a in arms]),
         "arms": arms,
         "note": "Absolutes are for the record. Scoring uses paired ratios measured in one "
                 "session — a baseline from another day is not comparable."}, indent=2) + "\n")
    Out.step(f"recorded goldens.json + baseline.json "
             f"({statistics.median([a['decodeTps'] for a in arms]):.2f} tok/s decode)")


def _paired(target, prompts, patch, pairs, tag):
    ratios, arms = [], []
    for i in range(pairs):
        b = run_arm(target, None, prompts, f"baseline (pair {i+1}/{pairs})", f"{tag}{i}-base")
        c = run_arm(target, patch, prompts, f"candidate (pair {i+1}/{pairs})", f"{tag}{i}-cand")
        arms += [b, c]
        ratios.append({"decode": c["decodeTps"] / b["decodeTps"],
                       "prefill": c["prefillTps"] / b["prefillTps"]})
        Out.info(f"pair {i+1}: decode x{ratios[-1]['decode']:.4f}  "
                 f"prefill x{ratios[-1]['prefill']:.4f}")
    return ratios, arms


def cmd_bench(args):
    target = load_target(args.target)
    prompts = prompt_set(target)
    gp = target["_dir"] / "goldens.json"
    goldens = json.loads(gp.read_text())["prompts"] if gp.exists() else {}

    Out.step("gate 1: config identity")
    wv = wheel_version()
    if wv != CONTRACT["substrate"]["wheel"]["version"]:
        die(f"installed vLLM is {wv}, pinned is {CONTRACT['substrate']['wheel']['version']} — "
            f"a different wheel is a different substrate and its scores are not comparable")
    Out.ok(f"vLLM {wv} matches the pin")
    Out.ok(f"model {model_fingerprint(target)[:16]}")

    ratios, arms = _paired(target, prompts, args.patch,
                           args.pairs or CONTRACT["scoring"]["pairs"], "p")
    dec = statistics.median(r["decode"] for r in ratios)
    pre = statistics.median(r["prefill"] for r in ratios)

    print()
    Out.step("gate 2: token identity")
    identity = True
    for arm in [a for a in arms if a["patch"]]:
        for r in arm["prompts"]:
            want = goldens.get(r["id"])
            if want and want["sha256"] != r["sha256"]:
                Out.fail(f"{r['id']}: output changed ({want['sha256'][:12]} -> {r['sha256'][:12]})")
                identity = False
    if identity:
        Out.ok(f"gate 2 ({len(prompts)}/{len(prompts)} identical)")

    Out.step("gate 4: speedup floors")
    floor = CONTRACT["scoring"]["floor"]
    floors = dec >= floor and pre >= floor
    for n, v in (("decode", dec), ("prefill", pre)):
        (Out.ok if v >= floor else Out.fail)(f"{n} speedup x{v:.4f} (floor {floor})")

    score = (dec ** CONTRACT["scoring"]["decodeExponent"]) * \
            (pre ** CONTRACT["scoring"]["prefillExponent"])
    rec = {"contractVersion": CONTRACT["contractVersion"], "target": target["_slug"],
           "engine": "vllm", "at": now_iso(), "node": os.uname().nodename,
           "wheel": wv, "patch": args.patch, "pairs": len(ratios), "ratios": ratios,
           "decodeSpeedup": round(dec, 5), "prefillSpeedup": round(pre, 5),
           "score": round(score, 5),
           "gates": {"configIdentity": True, "tokenIdentity": identity, "speedupFloors": floors},
           "promotable": bool(identity and floors and score > 1.0), "arms": arms}
    out = ROOT / "results" / \
        f"{target['_slug']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print()
    verdict = (f"{Out.GRN}SCORE {score:.4f}{Out.OFF}" if rec["promotable"]
               else f"{Out.RED}REJECTED{Out.OFF}")
    print(f"{Out.BOLD}{target['_slug']}{Out.OFF}: decode x{dec:.4f}  "
          f"prefill x{pre:.4f}  ->  {verdict}")
    print(f"    {out.relative_to(ROOT)}")
    if rec["promotable"]:
        print(f"    {Out.DIM}gate 3 (held-out identity AND speedup) runs on the referee's "
              f"node — pass --claimed-speedup {dec:.5f}.{Out.OFF}")
    sys.exit(0 if rec["promotable"] else 1)


def cmd_heldout(args):
    """Gate 3: identity AND speedup on prompts nobody has seen.

    The speedup half is what makes this arena safe. The submission is arbitrary
    Python, so a patch could cache completions and return identical tokens at
    absurd speed. A cache cannot hit a freshly generated prompt — but only a
    TIMED held-out arm notices.
    """
    target = load_target(args.target)
    seed = args.seed or os.urandom(16).hex()
    prompts = generate_heldout(target, seed, args.count)
    Out.step(f"gate 3: held-out identity + speedup ({args.count} generated prompts)")
    Out.info(f"seed {seed} — recorded, so these can be regenerated to audit this "
             f"verification, but not before it")

    ratios, arms = _paired(target, prompts, args.patch, args.pairs, "h")
    dec = statistics.median(r["decode"] for r in ratios)
    pre = statistics.median(r["prefill"] for r in ratios)

    base_rows = {r["id"]: r for a in arms if not a["patch"] for r in a["prompts"]}
    mismatches = []
    for a in arms:
        if not a["patch"]:
            continue
        for r in a["prompts"]:
            b = base_rows.get(r["id"])
            if b and b["sha256"] != r["sha256"]:
                mismatches.append({"id": r["id"], "baseSha256": b["sha256"],
                                   "candidateSha256": r["sha256"]})
                Out.fail(f"{r['id']}: output differs ({b['sha256'][:12]} -> {r['sha256'][:12]})")
    identity = not mismatches
    if identity:
        Out.ok(f"held-out identity ({len(prompts)}/{len(prompts)} identical on unseen prompts)")

    tol = next(g for g in CONTRACT["gates"] if g["id"] == "held-out-identity-and-speedup")["tolerance"]
    claimed = args.claimed_speedup
    generalizes = True
    if claimed and claimed > 1.0:
        need = 1.0 + (claimed - 1.0) * tol
        generalizes = dec >= need
        (Out.ok if generalizes else Out.fail)(
            f"held-out decode x{dec:.4f} vs claimed x{claimed:.4f} "
            f"(needs >= x{need:.4f}, {tol:.0%} of the claimed gain)")
        if not generalizes:
            Out.info("a win that does not reproduce on unseen prompts is a lookup table, "
                     "not an optimization")
    else:
        Out.warn("no --claimed-speedup given; speed generalization NOT checked — this is the "
                 "half of gate 3 that catches a memoizing patch")

    passed = identity and generalizes
    rec = {"contractVersion": CONTRACT["contractVersion"],
           "gate": "held-out-identity-and-speedup", "target": target["_slug"], "engine": "vllm",
           "at": now_iso(), "node": os.uname().nodename, "wheel": wheel_version(),
           "patch": args.patch, "referee": args.referee, "seed": seed,
           "promptCount": args.count,
           "heldOutDecodeSpeedup": round(dec, 5), "heldOutPrefillSpeedup": round(pre, 5),
           "claimedDecodeSpeedup": claimed, "tolerance": tol,
           "identical": identity, "generalizes": generalizes, "passed": passed,
           "mismatches": mismatches,
           "regenerate": (f"python3 harness/arena.py heldout --target {target['_slug']} "
                          f"--patch {args.patch} --seed {seed} --count {args.count}"),
           "arms": arms}
    out = ROOT / "results" / target["_slug"] / \
        f"heldout-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    Out.info(f"written to {out.relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


# ------------------------------------------------------------- leaderboard

def leaderboard_path(target):
    return ROOT / "results" / target["_slug"] / "leaderboard.json"


def load_leaderboard(target, create=False):
    """The ordered promotion chain for one target.

    This file, not results/*.json, is what a frontier chart is drawn from. A
    bench record says "x1.04 against whatever the incumbent was that day";
    only an ordered chain can say "x1.31 against the original baseline", and
    that chain cannot be reconstructed after the fact -- which is why it is
    written from the first promotion rather than added once there is something
    to plot.
    """
    path = leaderboard_path(target)
    if path.exists():
        return json.loads(path.read_text())
    if not create:
        die(f"no leaderboard for {target['_slug']} -- run `promote` on a passing record first")
    base_path = target["_dir"] / "baseline.json"
    if not base_path.exists():
        die(f"no baseline.json for {target['_slug']} -- run `baseline` first")
    base = json.loads(base_path.read_text())
    return {
        "contractVersion": CONTRACT["contractVersion"],
        "target": target["_slug"],
        "targetName": target.get("name", target["_slug"]),
        "recipe": target.get("recipe"),
        "engine": "vllm",
        "hardware": CONTRACT["hardware"]["name"],
        "vendorPin": f"vllm {CONTRACT['substrate']['wheel']['version']}",
        "repoUrl": CONTRACT.get("repoUrl"),
        "origin": {
            "recordedAt": base["recordedAt"],
            "node": base["node"],
            "buildInfo": base.get("wheel"),
            "decodeTps": base["decodeTps"],
            "prefillTps": base["prefillTps"],
            "note": "The frontier is measured against THIS. Every cumulative "
                    "number on the chart is a product of paired ratios back to here.",
        },
        "current": {
            "decodeTps": base["decodeTps"],
            "prefillTps": base["prefillTps"],
            "cumulativeDecodeSpeedup": 1.0,
            "cumulativePrefillSpeedup": 1.0,
            "cumulativeScore": 1.0,
            "promotionCount": 0,
        },
        "promotions": [],
    }


def cmd_promote(args):
    """Promote a passing bench record onto the frontier.

    Deliberately a separate step from `bench`. Gate 3 (held-out prompts) and
    gate 5 (beat the incumbent) are the referee's, not the contributor's, so
    the thing that writes history is run by whoever ran those.
    """
    target = load_target(args.target)
    record = json.loads(Path(args.record).read_text())

    if record["target"] != target["_slug"]:
        die(f"record is for target '{record['target']}', not '{target['_slug']}'")
    if record["contractVersion"] != CONTRACT["contractVersion"]:
        die(f"record is contractVersion {record['contractVersion']}, contract is "
            f"{CONTRACT['contractVersion']} -- scores are only comparable within one version")
    if not record.get("promotable") and not args.force:
        failed = [k for k, v in record["gates"].items() if not v]
        die(f"record did not pass its own gates ({', '.join(failed) or 'score <= 1.0'})")
    # Gate 3 must be BACKED BY A RECORD, not asserted. A boolean flag is a
    # promise; a record names the seed, the node and the changed files, so the
    # verification can be reproduced by anyone later. "Verified" is granted, not
    # owed -- and a claim nobody can re-check is not a grant.
    held_out = None
    if args.held_out_record:
        held_out = json.loads(Path(args.held_out_record).read_text())
        if held_out.get("gate") != "held-out-identity-and-speedup":
            die(f"{args.held_out_record} is not a held-out verification record")
        if held_out["target"] != target["_slug"]:
            die(f"held-out record is for target '{held_out['target']}'")
        if not held_out.get("passed"):
            die(f"held-out verification FAILED on {len(held_out.get('mismatches', []))} "
                f"prompt(s) — this candidate changes output on inputs it was not "
                f"tuned against, which is exactly what gate 3 exists to catch.")
        # The record has to describe THIS candidate. Verifying one diff and
        # promoting another is the obvious way to launder a failing submission.
        if held_out.get("patch") != record.get("patch"):
            die(f"held-out record verified patch '{held_out.get('patch')}' but the bench "
                f"record is for '{record.get('patch')}'. Verifying one submission and "
                f"promoting another is the obvious way to launder a failing one.")
        # vLLM-specific: gate 3 also proves the win GENERALIZES. A patch here is
        # arbitrary Python and could memoize, so identity alone is not enough.
        if not held_out.get("generalizes", False):
            die("held-out speedup did not generalize to unseen prompts. A win that only "
                "appears on prompts the submitter chose is a lookup table, not an "
                "optimization.")
    elif not args.force:
        die("gate 3 (held-out token identity) has no verification record.\n"
            "    Run it on the referee's node:\n"
            f"      python3 harness/arena.py heldout --target {target['_slug']}\n"
            "    then pass --held-out-record results/<target>/heldout-<stamp>.json.\n"
            "    This is the gate that catches a kernel which reassociated its way "
            "into a different argmax on everything except the prompts it was tuned "
            "against -- promoting without it is the one shortcut that silently "
            "corrupts the corpus.")

    lb = load_leaderboard(target, create=True)
    seq = len(lb["promotions"]) + 1

    # Cumulative = product of paired ratios back to the origin. Each ratio was
    # measured against the incumbent of its day, so the chain multiplies. It
    # never re-derives from absolute tok/s: absolutes are not comparable across
    # days on this fleet (24.05 -> 20.09 observed overnight).
    prev = lb["current"]
    cum_decode = prev["cumulativeDecodeSpeedup"] * record["decodeSpeedup"]
    cum_prefill = prev["cumulativePrefillSpeedup"] * record["prefillSpeedup"]
    cum_score = (cum_decode ** CONTRACT["scoring"]["decodeExponent"]) * \
                (cum_prefill ** CONTRACT["scoring"]["prefillExponent"])

    promotion = {
        "seq": seq,
        "promotedAt": now_iso(),
        "measuredAt": record["at"],
        "author": {
            "handle": args.author,
            "name": args.author_name or args.author,
            "model": args.model,
        },
        "note": Path(args.note).read_text().strip() if args.note and Path(args.note).exists() else (args.note or ""),
        "submissionUrl": args.url,
        "scope": args.scope,
        "alsoMoved": args.also_moved or [],
        "changedFiles": record.get("changedFiles", []),
        "patch": record.get("patch"),
        "kernels": sorted({Path(f).name for f in record.get("changedFiles", [])}) or [record.get("patch") or ""],
        "vsIncumbent": {
            "decodeSpeedup": record["decodeSpeedup"],
            "prefillSpeedup": record["prefillSpeedup"],
            "score": record["score"],
            "pairs": record["pairs"],
        },
        "cumulative": {
            "decodeSpeedup": round(cum_decode, 5),
            "prefillSpeedup": round(cum_prefill, 5),
            "score": round(cum_score, 5),
            "percentFaster": round((cum_decode - 1) * 100, 2),
            # Absolutes projected from the origin through the ratio chain, so a
            # leaderboard row can show tok/s without anyone re-measuring an old
            # submission on today's thermals.
            "decodeTps": round(lb["origin"]["decodeTps"] * cum_decode, 3),
            "prefillTps": round(lb["origin"]["prefillTps"] * cum_prefill, 2),
            # What THIS solver added, in percentage points of the headline. Not
            # the same as (decodeSpeedup - 1): a x1.01 win on top of a x1.37
            # frontier adds 1.4pp, not 1.0pp. This is the number the leaderboard
            # shows in green, and getting it from the ratio would understate
            # every late win.
            "addedPct": round((cum_decode - prev["cumulativeDecodeSpeedup"]) * 100, 2),
        },
        "verification": {
            "node": record["node"],
            "gates": record["gates"],
            "heldOutVerified": bool(held_out and held_out.get("passed")),
            "heldOutGeneralizes": bool(held_out and held_out.get("generalizes")),
            "heldOut": {
                "seed": held_out["seed"],
                "promptCount": held_out["promptCount"],
                "at": held_out["at"],
                "node": held_out["node"],
                "regenerate": held_out["regenerate"],
                "heldOutDecodeSpeedup": held_out.get("heldOutDecodeSpeedup"),
            } if held_out else None,
            "referee": args.referee,
        },
    }
    lb["promotions"].append(promotion)
    lb["current"] = {
        # Absolute tok/s is projected from the origin through the ratio chain,
        # not read off today's run, so the series stays internally consistent.
        "decodeTps": round(lb["origin"]["decodeTps"] * cum_decode, 3),
        "prefillTps": round(lb["origin"]["prefillTps"] * cum_prefill, 2),
        "cumulativeDecodeSpeedup": round(cum_decode, 5),
        "cumulativePrefillSpeedup": round(cum_prefill, 5),
        "cumulativeScore": round(cum_score, 5),
        "promotionCount": seq,
        "leaderHandle": args.author,
        "updatedAt": now_iso(),
    }

    path = leaderboard_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lb, indent=2) + "\n")
    Out.step(f"promoted #{seq} by {args.author}")
    Out.info(f"vs incumbent : decode x{record['decodeSpeedup']:.4f}  prefill x{record['prefillSpeedup']:.4f}")
    Out.info(f"cumulative   : decode x{cum_decode:.4f}  ({(cum_decode-1)*100:+.2f}% vs origin)")
    Out.info(f"frontier     : {lb['current']['decodeTps']} tok/s decode")
    Out.info(f"written to   : {path.relative_to(ROOT)}")


def cmd_leaderboard(args):
    target = load_target(args.target)
    lb = load_leaderboard(target, create=args.init)
    if args.init and not leaderboard_path(target).exists():
        # A target with zero promotions still publishes a leaderboard: the site
        # needs the origin to draw the flat baseline the frontier departs from,
        # and an empty chart is a truthful chart.
        leaderboard_path(target).parent.mkdir(parents=True, exist_ok=True)
        leaderboard_path(target).write_text(json.dumps(lb, indent=2) + "\n")
        Out.info(f"initialized {leaderboard_path(target).relative_to(ROOT)}")
    cur = lb["current"]
    print(f"\n{Out.BOLD}{lb['targetName']}{Out.OFF}  ({lb['target']})")
    print(f"  recipe    : {lb.get('recipe') or '-'}")
    print(f"  origin    : {lb['origin']['decodeTps']} tok/s decode, "
          f"{lb['origin']['prefillTps']} tok/s prefill  ({lb['origin']['recordedAt'][:10]})")
    print(f"  frontier  : {cur['decodeTps']} tok/s decode  "
          f"{Out.GRN}{(cur['cumulativeDecodeSpeedup']-1)*100:+.2f}%{Out.OFF} "
          f"over {cur['promotionCount']} promotion(s)\n")
    if not lb["promotions"]:
        print(f"  {Out.DIM}no promotions yet -- the frontier is the baseline{Out.OFF}\n")
        return
    print(f"  {'#':>2}  {'when':<11} {'who':<18} {'vs inc':>8} {'cumulative':>11}  kernels")
    for p in lb["promotions"]:
        print(f"  {p['seq']:>2}  {p['promotedAt'][:10]:<11} {p['author']['handle'][:18]:<18} "
              f"x{p['vsIncumbent']['decodeSpeedup']:.4f} {p['cumulative']['percentFaster']:>+10.2f}%  "
              f"{', '.join(p['kernels'][:3]) or '-'}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="record goldens + baseline (no patch)")
    b.add_argument("--target", required=True)
    b.add_argument("--repeats", type=int, default=4,
                   help="boots. The first is discarded as compile-cache warmup, and the "
                        "rest screen each prompt for cross-boot stability, so 4 gives 3 "
                        "measured boots.")
    b.add_argument("--no-discard-first", dest="discard_first", action="store_false",
                   help="keep boot 0 (only correct if VLLM_CACHE_ROOT is already warm)")
    b.set_defaults(discard_first=True)
    b.set_defaults(fn=cmd_baseline)

    w = sub.add_parser("warm", help="compile every prompt shape's kernels into the "
                                    "on-disk cache; required before baselining a TP target")
    w.add_argument("--target", required=True)
    w.add_argument("--patch", default=None)
    w.add_argument("--timeout", type=int, default=3600,
                   help="per-request timeout; generous on purpose, a cold TileLang "
                        "compile under TP can take minutes and must not be killed")
    w.set_defaults(fn=cmd_warm)

    n = sub.add_parser("bench", help="paired patched-vs-baseline run with gates 1, 2, 4")
    n.add_argument("--target", required=True)
    n.add_argument("--patch", required=True, help="directory name under patches/")
    n.add_argument("--pairs", type=int, default=None)
    n.set_defaults(fn=cmd_bench)

    p_ = sub.add_parser("promote", help="referee: append a passing record to the frontier")
    p_.add_argument("--target", required=True)
    p_.add_argument("--record", required=True, help="path to a results/*.json bench record")
    p_.add_argument("--author", required=True, help="GitHub handle of the contributor")
    p_.add_argument("--author-name", default=None)
    p_.add_argument("--model", default=None, help="the model that wrote it, if any")
    p_.add_argument("--note", default=None, help="one-line summary, or a path to a note file")
    p_.add_argument("--url", default=None, help="PR or submission URL")
    p_.add_argument("--scope", choices=["engine-general", "model-specific"], default="model-specific")
    p_.add_argument("--also-moved", nargs="*", default=None)
    p_.add_argument("--referee", default=None)
    p_.add_argument("--held-out-record", default=None,
                    help="path to a passing `heldout` record (gate 3: identity AND speedup)")
    p_.add_argument("--force", action="store_true")
    p_.set_defaults(fn=cmd_promote)

    l = sub.add_parser("leaderboard", help="print the frontier and its promotion chain")
    l.add_argument("--target", required=True)
    l.add_argument("--init", action="store_true", help="create it from baseline.json if absent")
    l.set_defaults(fn=cmd_leaderboard)

    h = sub.add_parser("heldout", help="gate 3: identity AND speedup on generated prompts")
    h.add_argument("--target", required=True)
    h.add_argument("--patch", required=True)
    h.add_argument("--count", type=int, default=6)
    h.add_argument("--pairs", type=int, default=1)
    h.add_argument("--seed", default=None, help="omit for fresh; pass a recorded seed to audit")
    h.add_argument("--claimed-speedup", type=float, default=None,
                   help="decodeSpeedup from the bench record, so generalization can be checked")
    h.add_argument("--referee", default=None)
    h.set_defaults(fn=cmd_heldout)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
