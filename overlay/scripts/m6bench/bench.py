"""M6 benchmark: one backend + one model per invocation, run ON the M6 so that powermetrics and the request
timestamps share one clock.

    python bench.py --backend gpu|ane --model bonsai|ornith --runs 3 --gen 256 --out DIR [--draft]

It starts the server, waits for health, does one warm-up request, then for each prompt size (512, 2048 tokens)
runs N streamed requests (thinking off, temperature 0). Prefill and decode tokens/s come from the servers' own
timing fields (llama-server `timings`; Forge `/health` prefill / decode stats). If `sudo -n powermetrics` works,
it samples CPU+GPU+ANE power for the whole run and integrates energy inside each request's prefill window
[send, first token] and decode window [first token, end]. Writes DIR/<model>-<backend>.json and the raw logs.
"""
import argparse
import json
import os
import plistlib
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
ENV = os.environ.get
# Machine layout (override with env): server bind address, llama.cpp build, GGUF files, Forge bundles, text corpus.
HOST = ENV("M6BENCH_HOST", "127.0.0.1")
PORT = {"gpu": int(ENV("M6BENCH_GPU_PORT", "8080")), "ane": int(ENV("M6BENCH_ANE_PORT", "8765"))}
FORGE_DIR = Path(ENV("FORGE_DIR", str(Path(__file__).resolve().parents[2])))
LLAMA_DIR = Path(ENV("LLAMA_CPP_DIR", str(HOME / "models/llamacpp")))
GGUF_DIR = Path(ENV("GGUF_DIR", str(HOME / "models/gguf")))
MODELS_DIR = Path(ENV("FORGE_MODELS_DIR", str(HOME / "Models")))
GGUF = {"bonsai": GGUF_DIR / "Ternary-Bonsai-2-27B-PTQ1_0.gguf", "ornith": GGUF_DIR / "Ornith-1.5-9B-Q4_K_M.gguf"}
FORGE = {"bonsai": MODELS_DIR / "forge-bonsai2", "ornith": MODELS_DIR / "forge-ornith15"}
FORGE_DRAFT = {"bonsai": MODELS_DIR / "forge-bonsai2-taps"}  # target build with the drafter's feature taps
TOKENIZER = {"bonsai": MODELS_DIR / "forge-bonsai2/model", "ornith": MODELS_DIR / "forge-ornith15/model"}
TEXT = Path(ENV("WIKITEXT", str(MODELS_DIR / "wikitext/wiki2_test.txt")))
PM_INTERVAL_MS = 50


def server_cmd(backend, model, draft):
    if backend == "gpu":
        return ([str(LLAMA_DIR / "llama-server"), "-m", str(GGUF[model]), "--host", HOST,
                 "--port", str(PORT["gpu"]), "-ngl", "99", "-c", "8192", "-fa", "on", "--jinja", "-np", "1"],
                LLAMA_DIR)
    b = FORGE_DRAFT[model] if draft else FORGE[model]
    cmd = [str(FORGE_DIR / ".venv/bin/python"), "forge.py", "serve", "--runtime", "coreai",
           "--model", str(b / "model"), "--build", str(b / "coreai"), "--ctx", "8192",
           "--host", HOST, "--port", str(PORT["ane"])]
    if draft:
        cmd += ["--draft", str(b / "drafter/dflash2_lut4_gptq.aimodel"), "--drafter", str(b / "drafter")]
    else:
        cmd += ["--plain"]
    return cmd, FORGE_DIR


def http(url, body=None, timeout=3600):
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def wait_health(base, proc, limit=900):
    t = time.time()
    while time.time() - t < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with http(base + "/health", timeout=3) as r:
                if r.status == 200 and json.load(r).get("status") in ("ok", None):
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise RuntimeError("health timeout")


def make_prompts(model):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER[model] / "tokenizer.json"))
    text = TEXT.read_text()
    ids = tok.encode(text).ids
    head = "Read the following text, then write a long and detailed summary of it.\n\n"
    out = {}
    for n in (512, 2048):
        out[n] = head + tok.decode(ids[1000:1000 + n - 40])
    return out


def stream(base, prompt, gen, run_tag):
    body = {"model": "x", "messages": [{"role": "user", "content": f"[run {run_tag}] {prompt}"}],
            "max_tokens": gen, "temperature": 0, "stream": True, "cache_prompt": False,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t_send = time.time()
    t_first, text, last = None, "", None
    with http(base + "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            last = ev if ("timings" in ev or "usage" in ev) else last
            for ch in ev.get("choices") or []:
                d = (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content")
                if d:
                    if t_first is None:
                        t_first = time.time()
                    text += d
    return t_send, t_first, time.time(), text, last or {}


MACMON = Path(ENV("MACMON", str(HOME / ".local/bin/macmon")))


def pm_start(path):
    """Power sampler: macmon (IOReport, no sudo) if present, else sudo powermetrics. Returns ((proc, file, t0, kind), why)."""
    if MACMON.exists():
        f = open(path, "w")
        p = subprocess.Popen([str(MACMON), "pipe", "-i", str(PM_INTERVAL_MS)], stdout=f, stderr=subprocess.DEVNULL)
        return (p, f, time.time(), "macmon"), None
    probe = subprocess.run(["sudo", "-n", "/usr/bin/powermetrics", "-n", "1", "-i", "50", "--samplers", "cpu_power"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        return None, probe.stderr.strip()
    f = open(path, "wb")
    p = subprocess.Popen(["sudo", "-n", "/usr/bin/powermetrics", "-i", str(PM_INTERVAL_MS), "-f", "plist",
                          "--samplers", "cpu_power,gpu_power,ane_power"], stdout=f, stderr=subprocess.DEVNULL)
    return (p, f, time.time(), "powermetrics"), None


def macmon_samples(path):
    """[(t_begin, t_end, sys_W*1000, cpu, gpu, ane in mW)] from macmon JSON lines; sample i covers (t_{i-1}, t_i]."""
    from datetime import datetime
    out, prev = [], None
    for line in Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        t = datetime.fromisoformat(d["timestamp"]).timestamp()
        if prev is not None:
            out.append((prev, t, d["sys_power"] * 1000, d["cpu_power"] * 1000, d["gpu_power"] * 1000,
                        d["ane_power"] * 1000))
        prev = t
    return out


def pm_samples(path, t_start):
    """[(t_begin, t_end, combined_mW, cpu_mW, gpu_mW, ane_mW)] with times on this machine's clock."""
    raw = Path(path).read_bytes().split(b"\x00")
    out, t = [], None
    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            d = plistlib.loads(chunk)
        except Exception:  # noqa: BLE001
            continue
        dur = d["elapsed_ns"] / 1e9
        ts = d.get("timestamp")
        end = ts.timestamp() if ts is not None and abs(ts.timestamp() - time.time()) < 86400 else None
        if end is None:
            t = (t or t_start) + dur
            end = t
        pr = d.get("processor", {})
        cpu = pr.get("cpu_power", 0.0)
        gpu = pr.get("gpu_power", d.get("gpu", {}).get("gpu_power", 0.0))
        ane = pr.get("ane_power", 0.0)
        comb = pr.get("combined_power", cpu + gpu + ane)
        out.append((end - dur, end, comb, cpu, gpu, ane))
    return out


def energy(samples, a, b):
    """Joules (combined, cpu, gpu, ane) of the samples overlapping [a, b]."""
    tot = [0.0, 0.0, 0.0, 0.0]
    for s0, s1, *p in samples:
        ov = min(s1, b) - max(s0, a)
        if ov > 0:
            for k in range(4):
                tot[k] += p[k] / 1000.0 * ov
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("gpu", "ane"), required=True)
    ap.add_argument("--model", choices=("bonsai", "ornith"), required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--build", help="Forge bundle dir (model/, coreai/) overriding the default for --model")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}-{a.backend}" + ("-draft" if a.draft else "") + (f"-{Path(a.build).name}" if a.build else "")
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-f", "qwen38_server"], capture_output=True)
    time.sleep(3)
    if a.build:
        (FORGE_DRAFT if a.draft else FORGE)[a.model] = Path(a.build)
    cmd, cwd = server_cmd(a.backend, a.model, a.draft)
    log = open(a.out / f"server-{tag}.log", "w")
    env = dict(os.environ, FORGE_FOREIGN_DRAFTER="1") if a.draft else None
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    base = f"http://{HOST}:{PORT[a.backend]}"
    result = {"tag": tag, "cmd": cmd, "runs": []}
    pm = None
    try:
        wait_health(base, proc)
        if a.backend == "ane":
            with http(base + "/health") as r:
                result["health"] = json.load(r)
            b = FORGE_DRAFT[a.model] if a.draft else FORGE[a.model]
            insp = [str(FORGE_DIR / ".venv/bin/python"), str(FORGE_DIR / "coreai/inspect_coreai_cache.py"),
                    "--model-dir", str(b / "coreai"), "--strict"]
            if a.draft:
                insp += ["--drafter", str(b / "drafter/dflash2_lut4_gptq.aimodel")]
            pl = subprocess.run(insp, capture_output=True, text=True)
            result["placement"] = {p_["package"].rsplit("/", 1)[-1]: p_.get("status")
                                   for p_ in json.loads(pl.stdout)["packages"]}
            print("placement:", json.dumps(result["placement"]), flush=True)
            if pl.returncode != 0:
                raise RuntimeError(f"not every package is fully on the ANE: {result['placement']}")
        prompts = make_prompts(a.model)
        stream(base, "Say hello.", 16, "warmup")
        time.sleep(5)
        pm, why = pm_start(a.out / f"pm-{tag}.plist")
        result["powermetrics"] = pm[3] if pm else f"unavailable: {why}"
        time.sleep(3)
        for n, prompt in prompts.items():
            for i in range(a.runs):
                t_send, t_first, t_end, text, last = stream(base, prompt, a.gen, f"{n}-{i}")
                rec = {"prompt_size": n, "run": i, "t_send": t_send, "t_first": t_first, "t_end": t_end,
                       "text_head": text[:160]}
                if a.backend == "gpu":
                    tm = last.get("timings", {})
                    rec.update(prompt_tokens=tm.get("prompt_n"), prefill_tps=tm.get("prompt_per_second"),
                               gen_tokens=tm.get("predicted_n"), decode_tps=tm.get("predicted_per_second"))
                else:
                    with http(base + "/health") as r:
                        h = json.load(r)
                    rec.update(prompt_tokens=h["prefill"]["tokens"], cached=h["prefill"]["cached_tokens"],
                               prefill_tps=h["prefill"]["tokens_per_second"], gen_tokens=h["decode"]["tokens"],
                               decode_tps=h["decode"]["tokens_per_second"],
                               accept_pct=h["decode"].get("draft_accept_percent"),
                               tokens_per_call=h["decode"].get("tokens_per_call"))
                result["runs"].append(rec)
                print(json.dumps({k: rec[k] for k in ("prompt_size", "run", "prompt_tokens", "prefill_tps",
                                                       "gen_tokens", "decode_tps")}), flush=True)
                time.sleep(3)
    finally:
        if pm:
            pm[0].send_signal(signal.SIGINT if pm[3] == "powermetrics" else signal.SIGTERM)
            time.sleep(1)
            pm[0].terminate()
            pm[1].close()
        os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(3)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    if pm:
        samples = (macmon_samples(a.out / f"pm-{tag}.plist") if pm[3] == "macmon"
                   else pm_samples(a.out / f"pm-{tag}.plist", pm[2]))
        result["power_source"] = pm[3] + (" (sys_power = whole-machine SMC power; cpu/gpu/ane = IOReport rails)"
                                          if pm[3] == "macmon" else " (combined_power = CPU+GPU+ANE)")
        idle = [x for x in samples if x[1] < result["runs"][0]["t_send"] - 1]
        result["idle_W"] = (sum(x[2] for x in idle) / len(idle) / 1000) if idle else None
        result["pm_samples"] = len(samples)
        for rec in result["runs"]:
            ep = energy(samples, rec["t_send"], rec["t_first"])
            ed = energy(samples, rec["t_first"], rec["t_end"])
            rec["prefill_J"], rec["decode_J"] = ep, ed
            rec["J_per_prompt_token"] = ep[0] / rec["prompt_tokens"]
            rec["J_per_gen_token"] = ed[0] / max(rec["gen_tokens"], 1)
    summ = {}
    for n in (512, 2048):
        rs = [r for r in result["runs"] if r["prompt_size"] == n]
        s = {k: statistics.median([r[k] for r in rs]) for k in ("prefill_tps", "decode_tps", "prompt_tokens",
                                                                 "gen_tokens")}
        if pm:
            s["J_per_prompt_token"] = statistics.median([r["J_per_prompt_token"] for r in rs])
            s["J_per_gen_token"] = statistics.median([r["J_per_gen_token"] for r in rs])
        summ[n] = s
    result["median"] = summ
    (a.out / f"{tag}.json").write_text(json.dumps(result, indent=1))
    print("MEDIAN", tag, json.dumps(summ), flush=True)


if __name__ == "__main__":
    sys.exit(main())
