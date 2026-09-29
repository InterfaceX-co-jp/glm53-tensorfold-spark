"""Host-only tests of the launcher's ops paths (scripts/serve.sh, scripts/canary.py, scripts/xid.py).

No GPU, no Docker, no second node: ``docker``, ``ssh``, ``nvidia-smi`` and ``journalctl`` are shell fakes on PATH
(``ssh`` runs the remote command locally; both "nodes" share the fake Docker state), and the engine is a small HTTP
server answering the OpenAI routes the canary and the watchdog use.

    python -m pytest -q tests/test_serve_ops.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import canary  # noqa: E402
import xid  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("curl") is None
                                or shutil.which("flock") is None, reason="needs bash, curl and flock")

GOOD = {
    "capital": "Paris",
    "count": "1 2 3 4 5 6 7 8 9 10 11 12",
    "code": "def add(a, b):\n    return a + b",
}


class Engine:
    """What the fake server answers: per-probe content, rounds, health code, metrics text."""

    def __init__(self) -> None:
        self.replies = dict(GOOD)
        self.tokens_per_round = 4.0
        self.health = (200, {"ok": True})
        self.metrics = ""
        self.requests: list[dict] = []

    def reply(self, body: dict) -> dict:
        self.requests.append(body)
        text = body["messages"][0]["content"]
        content = "OK"
        for key, needle in (("capital", "capital"), ("count", "Count"), ("code", "add(a, b)")):
            if needle in text:
                content = self.replies[key]
        tokens = max(len(content.split()) * 2, 2)
        rounds = max(int((tokens - 1) / self.tokens_per_round), 1)
        return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(text.split()), "completion_tokens": tokens},
                "tensorfold": {"rounds": rounds, "decode_s": 0.05 * rounds, "prefill_s": 0.1}}


@pytest.fixture
def engine():
    state = Engine()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, data, ctype="application/json"):
            raw = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send(200, {"data": [{"id": "fake"}]})
            elif self.path == "/health":
                self._send(*state.health)
            elif self.path == "/metrics":
                self._send(200, state.metrics.encode(), "text/plain")
            else:
                self._send(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(200, state.reply(body))

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state.port = srv.server_address[1]
    state.base = f"http://127.0.0.1:{state.port}"
    yield state
    srv.shutdown()
    srv.server_close()


# -- canary.py ---------------------------------------------------------------------------------------------------

def test_canary_passes_healthy_engine(engine):
    report = canary.check(engine.base, "fake", min_tpr=1.3)
    assert report["ok"], report["problems"]
    assert report["tokens_per_round"] > 1.3
    # a nonce keeps a session-cache hit from answering the probe
    assert all(r["messages"][0]["content"].startswith("[canary ") for r in engine.requests)
    assert all(r["temperature"] == 0 and r["chat_template_kwargs"] == {"enable_thinking": False}
               for r in engine.requests)


@pytest.mark.parametrize("key,reply,why", [
    ("capital", "", "empty reply"),
    ("capital", "Lyon", "expected"),
    ("capital", "Paris Paris Paris Paris Paris Paris Paris", "the same word"),
    ("code", "the the of of the the of of and and the the of of and", "repetitive"),
    ("capital", "Paris �", "U+FFFD"),
    ("count", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "one character"),
])
def test_canary_catches_degenerate_output(engine, key, reply, why):
    engine.replies[key] = reply
    report = canary.check(engine.base, min_tpr=1.3)
    assert not report["ok"]
    assert any(why in p for p in report["problems"]), report["problems"]


def test_canary_catches_dead_drafter(engine):
    engine.tokens_per_round = 1.0
    report = canary.check(engine.base, min_tpr=1.3)
    assert not report["ok"] and any("drafter" in p for p in report["problems"])
    assert canary.check(engine.base, min_tpr=0)["ok"]           # NO_DRAFTS servers skip the check


def test_canary_min_tps(engine):
    assert not canary.check(engine.base, min_tps=1e6)["ok"]


def test_canary_cli_exit_codes(engine, capsys):
    assert canary.main(["--base", engine.base]) == 0
    engine.replies["capital"] = "Berlin"
    assert canary.main(["--base", engine.base]) == 1
    assert canary.main(["--base", "http://127.0.0.1:9", "--timeout", "2"]) == 2


def test_canary_warm_lengths(engine):
    assert canary.main(["--base", engine.base, "--warm-lengths", "100, 300"]) == 0
    warm = [r for r in engine.requests if r["messages"][0]["content"].startswith("[warmup ")]
    assert [len(r["messages"][0]["content"].split()) for r in warm] == [pytest.approx(100, abs=15),
                                                                         pytest.approx(300, abs=15)]
    assert all(r["max_tokens"] == 4 for r in warm)


def test_degenerate_short_answers_pass():
    assert canary.degenerate("Paris") is None
    assert canary.degenerate("OK.") is None
    assert canary.degenerate("x = 1\ny = 2\nprint(x + y)") is None


# -- xid.py ------------------------------------------------------------------------------------------------------

KLOG = """\
[  12.3] nvidia: loading out-of-tree module
Sep 27 10:00:01 dgx kernel: NVRM: Xid (PCI:000f:01:00): 43, pid=1234, name=python3, Ch 00000008
Sep 27 10:00:02 dgx kernel: NVRM: Xid (PCI:000f:01:00): 79, pid='<unknown>', name=<unknown>, GPU has fallen off the bus.
Sep 27 10:00:03 dgx kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=4321, name=tensorfold, Ch 0000001c, intr 0
"""


def test_xid_parse_and_classes():
    events = xid.parse(KLOG)
    assert [e["xid"] for e in events] == [43, 79, 31]          # the PCI address is not read as the Xid
    assert [e["class"] for e in events] == ["info", "fatal", "app"]
    assert events[0]["pid"] == 1234 and events[0]["process"] == "python3"
    assert xid.worst(events) == "fatal"
    assert xid.worst([]) is None
    assert xid.parse("NVRM: Xid (PCI:0000:01:00): 999, whatever")[0]["class"] == "app"   # unlisted: look at it


def run_xid(text, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/xid.py"), *args], input=text, text=True,
                          capture_output=True)


def test_xid_cli_fail_levels():
    assert run_xid(KLOG).returncode == 1
    only_app = "\n".join(KLOG.splitlines()[i] for i in (1, 3))
    assert run_xid(only_app).returncode == 0
    assert run_xid(only_app, "--fail-on", "app").returncode == 1
    assert run_xid("nothing here\n", "--fail-on", "any").returncode == 0


# -- serve.sh ----------------------------------------------------------------------------------------------------

FAKE_DOCKER = r"""#!/usr/bin/env bash
# fake docker: containers are files in $FAKE_STATE (content: true|false)
echo "docker $*" >> "$FAKE_STATE/calls"
case "$1" in
run)
    name=""; prev=""
    for a in "$@"; do [[ "$prev" == --name ]] && name=$a; prev=$a; done
    printf '%s\n' "$@" > "$FAKE_STATE/args.$name"
    echo "${FAKE_RUN_STATE:-true}" > "$FAKE_STATE/c.$name"; echo cid ;;
rm) for n in "${@:2}"; do [[ "$n" == -f ]] || rm -f "$FAKE_STATE/c.$n"; done ;;
inspect)
    fmt=$3; n=$4
    [[ -f "$FAKE_STATE/c.$n" ]] || { echo "Error: No such object: $n" >&2; exit 1; }
    case "$fmt" in
        *StartedAt*) date -u -d "@$(( $(date +%s) - ${FAKE_AGE:-5000} ))" +%Y-%m-%dT%H:%M:%S.000000000Z ;;
        *) cat "$FAKE_STATE/c.$n" ;;
    esac ;;
image) exit 0 ;;
logs) echo "log of ${@: -1}" ;;
ps) ls "$FAKE_STATE" | sed -n 's/^c\.//p' ;;
esac
"""

FAKE_SSH = r"""#!/usr/bin/env bash
# fake ssh: skip options and the target, run the command here
echo "ssh $*" >> "$FAKE_STATE/calls"
[[ "${FAKE_SSH_FAIL:-0}" == 1 ]] && exit 255
while [[ "$1" == -o ]]; do shift 2; done
shift
exec bash -c "$*"
"""

FAKE_SMI = r"""#!/usr/bin/env bash
case "$*" in
    *driver_version*) echo "${FAKE_DRIVER:-580.95.05}" ;;
    *) echo "${FAKE_GPU_PIDS:-}" ;;
esac
"""

FAKE_JOURNAL = r"""#!/usr/bin/env bash
[[ -f "$FAKE_STATE/klog" ]] && cat "$FAKE_STATE/klog"
exit 0
"""


@pytest.fixture
def kit(tmp_path, engine):
    """A copy of the repo's scripts with a fake config, fake tools on PATH and the fake engine's port."""

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "config").mkdir()
    for f in ("serve.sh", "canary.py", "xid.py"):
        shutil.copy(ROOT / "scripts" / f, repo / "scripts" / f)
    (repo / "config" / "tensorfold.env").write_text(
        "WORKER_SSH=fake-worker\nHEAD_IP=10.0.0.1\nNCCL_SOCKET_IFNAME=eth9\nNCCL_IB_HCA=fakehca0\n"
        "HEAD_HF=/tmp/hf\nWORKER_HF=/tmp/hf\nMODEL_PATH=/m\nIMAGE=fake:img\nSERVED_NAME=fake\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in (("docker", FAKE_DOCKER), ("ssh", FAKE_SSH), ("nvidia-smi", FAKE_SMI),
                       ("journalctl", FAKE_JOURNAL)):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GLM53_TF_", "NCCL_", "CANARY", "WATCH_",
                                                                    "MEM_GATE", "CONTEXT"))}
    env.update(PATH=f"{bin_}:{env['PATH']}", FAKE_STATE=str(state), PORT=str(engine.port),
               STATE_DIR=str(tmp_path / "ops"), XDG_STATE_HOME=str(tmp_path / "xdg"))

    def run(*args, timeout=60, **extra):
        e = dict(env, **{k: str(v) for k, v in extra.items()})
        return subprocess.run(["bash", str(repo / "scripts/serve.sh"), *args], env=e, text=True,
                              capture_output=True, timeout=timeout)

    kit = SimpleKit(run, state, tmp_path / "ops", engine)
    kit.cfg_no_head = tmp_path / "nohead.env"
    kit.cfg_no_head.write_text((repo / "config" / "tensorfold.env").read_text().replace("HEAD_IP=10.0.0.1\n", ""))
    return kit


class SimpleKit:
    def __init__(self, run, state, ops, engine):
        self.run, self.state, self.ops, self.engine = run, state, ops, engine

    def calls(self) -> str:
        p = self.state / "calls"
        return p.read_text() if p.exists() else ""

    def exists(self, name: str) -> bool:
        return (self.state / f"c.{name}").exists()


def test_start_defaults_unchanged(kit):
    """No ops knob set: no preflight, no canary, no log options, no NCCL extras, the original log format."""
    r = kit.run("start", NCCL_DEBUG="WARN")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[glm53-tf] ready after" in r.stdout
    assert "canary" not in r.stdout + r.stderr and "preflight" not in r.stdout
    assert not kit.engine.requests                                    # nothing sent to the engine
    assert kit.exists("glm53-tf-r0") and kit.exists("glm53-tf-r1")
    args = (kit.state / "args.glm53-tf-r0").read_text().split("\n")
    assert "--log-opt" not in args and "--stop-timeout" not in args
    assert not any(a.startswith("NCCL_DEBUG") for a in args)
    assert "NCCL_IB_HCA=fakehca0" in args and "GLOO_SOCKET_IFNAME=eth9" in args


def test_start_happy_path_runs_canary(kit):
    r = kit.run("start", CANARY="warn", PREFLIGHT="warn", LOG_MAX_SIZE="200m")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ready after" in r.stdout and "[canary] ok" in r.stderr
    assert kit.exists("glm53-tf-r0") and kit.exists("glm53-tf-r1")
    args = (kit.state / "args.glm53-tf-r0").read_text().split("\n")
    assert "max-size=200m" in args and "max-file=3" in args          # log rotation
    assert "preflight: warning: cannot read fakehca0" in r.stdout     # no RDMA sysfs here: warn, don't fail


def test_preflight_strict_blocks_start(kit):
    r = kit.run("start", PREFLIGHT="strict", CONFIG=str(kit.cfg_no_head))
    assert r.returncode == 1 and "not starting" in r.stdout
    r = kit.run("start", PREFLIGHT="warn", CONFIG=str(kit.cfg_no_head), CANARY="off")
    assert "starting anyway" in r.stdout


def test_no_home_no_state_dir(kit):
    r = kit.run("status", HOME="", XDG_STATE_HOME="", STATE_DIR="")
    assert r.returncode == 0, r.stderr


def test_start_refuses_when_gpu_busy(kit):
    r = kit.run("start", FAKE_GPU_PIDS="4242")
    assert r.returncode == 1 and "stop the other stack first" in r.stdout
    assert "docker run" not in kit.calls()


def test_preflight_missing_key(kit, tmp_path):
    cfg = tmp_path / "bad.env"
    cfg.write_text("WORKER_SSH=fake-worker\nHEAD_HF=/tmp/hf\nWORKER_HF=/tmp/hf\n")
    r = kit.run("preflight", CONFIG=str(cfg))
    assert r.returncode != 0 and "HEAD_IP is not set" in r.stdout


def test_preflight_driver_warnings(kit):
    r = kit.run("preflight")
    assert r.returncode == 0 and "driver" not in r.stdout
    r = kit.run("preflight", FAKE_DRIVER="590.10")
    assert r.returncode == 0 and "driver 590.x" in r.stdout


def test_nccl_passthrough(kit):
    r = kit.run("start", NCCL_IB_GID_INDEX="3", NCCL_DEBUG="WARN", NCCL_PASSTHROUGH=1)
    assert r.returncode == 0, r.stdout + r.stderr
    args = (kit.state / "args.glm53-tf-r1").read_text().split("\n")
    assert "NCCL_IB_GID_INDEX=3" in args and "NCCL_DEBUG=WARN" in args
    assert args.count("NCCL_IB_HCA=fakehca0") == 1                   # not passed twice


def test_cpuset(kit):
    """patches/0370: CPUSET -> docker run --cpuset-cpus on both ranks; HEAD_ / WORKER_ per node; unset: absent."""
    r = kit.run("start")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--cpuset-cpus" not in (kit.state / "args.glm53-tf-r0").read_text().split("\n")
    kit.run("stop")
    r = kit.run("start", CPUSET="5-9,15-19", WORKER_CPUSET="15-19")
    assert r.returncode == 0, r.stdout + r.stderr
    a0 = (kit.state / "args.glm53-tf-r0").read_text().split("\n")
    a1 = (kit.state / "args.glm53-tf-r1").read_text().split("\n")
    assert a0[a0.index("--cpuset-cpus") + 1] == "5-9,15-19"
    assert a1[a1.index("--cpuset-cpus") + 1] == "15-19"


def test_strict_canary_failure_stops_both(kit):
    kit.engine.replies["capital"] = "banana banana"
    r = kit.run("start", CANARY="strict")
    assert r.returncode == 1 and "failed the canary" in r.stdout, r.stdout
    assert not kit.exists("glm53-tf-r0") and not kit.exists("glm53-tf-r1")
    assert (kit.ops / "canary-fail-r0.log").exists()


def test_warn_canary_failure_keeps_serving(kit):
    kit.engine.replies["capital"] = "banana banana"
    r = kit.run("start", CANARY="warn")
    assert r.returncode == 0 and "still serving" in r.stdout
    assert kit.exists("glm53-tf-r0")


def test_rank_exit_fails_fast_and_retries(kit):
    r = kit.run("start", FAKE_RUN_STATE="false", START_ATTEMPTS=2, CANARY="off", PORT=9)
    assert r.returncode == 1
    assert r.stdout.count("rank 0 exited") == 2 and "tearing down and retrying" in r.stdout


def test_mem_gate(kit):
    r = kit.run("start", MEM_GATE_GIB=1, CANARY="off")
    assert r.returncode == 0 and ">= 1" in r.stdout
    r = kit.run("start", MEM_GATE_GIB=10**6, MEM_GATE_TIMEOUT=0, CANARY="off")
    assert r.returncode == 0 and "starting anyway" in r.stdout


def test_parallel_stop(kit):
    assert kit.run("start", CANARY="off").returncode == 0
    r = kit.run("stop")
    assert r.returncode == 0 and not kit.exists("glm53-tf-r0") and not kit.exists("glm53-tf-r1")


def test_start_lock_excludes_second_start(kit):
    kit.ops.mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen(["flock", str(kit.ops / "lock"), "sleep", "5"])
    try:
        import time
        time.sleep(0.3)
        r = kit.run("start", CANARY="off")
        assert r.returncode == 1 and "another start" in r.stdout
        r = kit.run("watch", "--once")
        assert r.returncode == 0 and "standing down" in r.stdout
    finally:
        holder.kill()
        holder.wait()


def test_xid_command(kit):
    (kit.state / "klog").write_text(KLOG)
    r = kit.run("xid")
    assert r.returncode == 1
    assert "head: Xid 79 [fatal]" in r.stdout and "worker: Xid 79 [fatal]" in r.stdout


def test_watch_absent_stands_down(kit):
    r = kit.run("watch", "--once")
    assert r.returncode == 0 and "stopped on purpose" in r.stdout


def test_watch_counts_bad_ticks_and_alerts_without_heal(kit, tmp_path):
    assert kit.run("start", CANARY="off").returncode == 0
    (kit.state / "c.glm53-tf-r1").write_text("false\n")            # rank 1 crashed
    alerts = tmp_path / "alerts"
    notify = tmp_path / "notify.sh"
    notify.write_text(f'#!/usr/bin/env bash\necho "$1" >> {alerts}\n')
    notify.chmod(0o755)
    for i in (1, 2):
        r = kit.run("watch", "--once", WATCH_ALERT=notify)
        assert r.returncode == 0 and f"bad tick {i} of 3: rank 1 false" in r.stdout
    r = kit.run("watch", "--once", WATCH_ALERT=notify)
    assert r.returncode == 1 and "WATCH_HEAL=0, not restarting" in r.stdout
    assert "rank 1 false" in alerts.read_text()
    assert kit.exists("glm53-tf-r0")                                 # nothing restarted


def test_watch_health_503_heals(kit, tmp_path):
    assert kit.run("start", CANARY="off").returncode == 0
    kit.engine.health = (503, {"ok": False, "fatal": "RuntimeError: NCCL"})
    for _ in range(2):
        assert kit.run("watch", "--once", WATCH_HEAL=1).returncode == 0
    r = kit.run("watch", "--once", WATCH_HEAL=1, CANARY="off")
    assert r.returncode == 1 and "restarting both ranks" in r.stdout and "NCCL" in r.stdout
    kit.engine.health = (200, {"ok": True})
    import time
    for _ in range(100):                                             # the heal runs detached
        log = kit.ops / "heal.log"
        if log.exists() and "ready after" in log.read_text():
            break
        time.sleep(0.2)
    assert "ready after" in (kit.ops / "heal.log").read_text()
    r = kit.run("watch", "--once", WATCH_HEAL=1)                     # healthy again, and rate-limited anyway
    assert r.returncode == 0


def test_watch_loading_grace(kit):
    assert kit.run("start", CANARY="off").returncode == 0
    r = kit.run("watch", "--once", FAKE_AGE=60, PORT=9)              # nothing listening yet, 60 s old
    assert r.returncode == 0 and "loading" in r.stdout
    r = kit.run("watch", "--once", FAKE_AGE=5000, PORT=9)            # past the grace: a bad tick
    assert "bad tick 1" in r.stdout


def test_watch_unreachable_worker_not_counted(kit):
    assert kit.run("start", CANARY="off").returncode == 0
    r = kit.run("watch", "--once", FAKE_SSH_FAIL=1)
    assert r.returncode == 0 and "worker unreachable" in r.stdout and "bad tick" not in r.stdout


def _metrics(tokens, rounds, reqs):
    return (f'tensorfold_completion_tokens_total{{model="fake"}} {tokens}\n'
            f'tensorfold_decode_rounds_total{{model="fake"}} {rounds}\n'
            f'tensorfold_requests_total{{model="fake"}} {reqs}\n')


def test_watch_drafter_rate_alert(kit):
    assert kit.run("start", CANARY="off").returncode == 0
    kit.engine.metrics = _metrics(1000, 300, 10)
    assert "drafter" not in kit.run("watch", "--once", WATCH_MIN_TPR=1.5, WATCH_TPR_ROUNDS=500).stdout   # seeds
    kit.engine.metrics = _metrics(1000 + 3000, 300 + 1000, 20)       # (3000 - 10) / 1000 = 2.99 a round
    assert "drafter" not in kit.run("watch", "--once", WATCH_MIN_TPR=1.5, WATCH_TPR_ROUNDS=500).stdout
    kit.engine.metrics = _metrics(4000 + 1010, 1300 + 1000, 30)      # (1010 - 10) / 1000 = 1.00
    r = kit.run("watch", "--once", WATCH_MIN_TPR=1.5, WATCH_TPR_ROUNDS=500)
    assert "ALERT: drafter: 1.00 tokens a round over 1000 rounds" in r.stdout
    kit.engine.metrics = _metrics(10, 5, 1)                          # server restarted: reseed, no alert
    assert "drafter" not in kit.run("watch", "--once", WATCH_MIN_TPR=1.5, WATCH_TPR_ROUNDS=500).stdout
