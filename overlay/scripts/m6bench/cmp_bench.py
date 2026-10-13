"""Engine-neutral M6 benchmark: the same prompts against any OpenAI-compatible server, measured from the client.

    python cmp_bench.py --name NAME --url http://HOST:PORT --out DIR [--runs 3] [--gen 256] [--model ID]

Prompts: the m6bench WikiText summaries (512 and 2048 prompt tokens, Bonsai tokenizer) and three code prompts.
Thinking off, temperature 0, streamed. Per request: TTFT, prefill tok/s = prompt_tokens / TTFT, decode tok/s =
(completion_tokens - 1) / (t_end - t_first). Energy from macmon (whole machine, sys_power) over [send, first token]
and [first token, end]. Server-reported prompt_tokens / completion_tokens are used when present, else the Bonsai
tokenizer count. The server must already be running; this script does not start or stop engines.
"""
import argparse
import json
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B  # noqa: E402  (make_prompts, macmon_samples, energy)

CODE = [
    "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling.",
    "Write a bash script that finds the ten largest files under a directory and prints their sizes.",
    "Write a TypeScript React component that shows a paginated list fetched from /api/items.",
]


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(B.TOKENIZER["bonsai"] / "tokenizer.json"))


def stream(url, model, prompt, gen, tag):
    body = {"model": model, "messages": [{"role": "user", "content": f"[run {tag}] {prompt}"}],
            "max_tokens": gen, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t_send, t_first, text, usage = time.time(), None, "", None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    text += piece
    return t_send, t_first, time.time(), text, usage or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    prompts = {f"wiki{n}": p for n, p in B.make_prompts("bonsai").items()}
    prompts.update({f"code{i}": p for i, p in enumerate(CODE)})
    stream(a.url, a.model, "Say hello.", 16, "warmup")
    time.sleep(3)
    pm_path = a.out / f"pm-{a.name}.jsonl"
    pmf = open(pm_path, "w")
    pm = subprocess.Popen([str(B.MACMON), "pipe", "-i", "50"], stdout=pmf, stderr=subprocess.DEVNULL)
    time.sleep(3)
    runs = []
    try:
        for key, prompt in prompts.items():
            for i in range(a.runs):
                t_send, t_first, t_end, text, usage = stream(a.url, a.model, prompt, a.gen, f"{key}-{i}")
                if t_first is None:  # no token came back (server error): record it and go on
                    print(json.dumps({"prompt": key, "run": i, "error": "no tokens", "usage": usage}), flush=True)
                    continue
                ptoks = usage.get("prompt_tokens") or len(tok.encode(prompt).ids)
                gtoks = usage.get("completion_tokens") or len(tok.encode(text).ids)
                rec = {"prompt": key, "run": i, "t_send": t_send, "t_first": t_first, "t_end": t_end,
                       "prompt_tokens": ptoks, "gen_tokens": gtoks, "ttft": t_first - t_send,
                       "prefill_tps": ptoks / (t_first - t_send),
                       "decode_tps": (gtoks - 1) / max(t_end - t_first, 1e-9), "text_head": text[:80]}
                runs.append(rec)
                print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()
                                  if k not in ("t_send", "t_first", "t_end", "text_head")}), flush=True)
                time.sleep(2)
    finally:
        pm.send_signal(signal.SIGTERM)
        time.sleep(1)
        pmf.close()
    samples = B.macmon_samples(pm_path)
    idle = [s for s in samples if s[1] < runs[0]["t_send"] - 0.5]
    for r in runs:
        ep, ed = B.energy(samples, r["t_send"], r["t_first"]), B.energy(samples, r["t_first"], r["t_end"])
        r["J_per_prompt_token"], r["J_per_gen_token"] = ep[0] / r["prompt_tokens"], ed[0] / max(r["gen_tokens"], 1)
        r["avg_W_decode"] = ed[0] / max(r["t_end"] - r["t_first"], 1e-9)
    med = {}
    for key in prompts:
        rs = [r for r in runs if r["prompt"] == key]
        if not rs:
            med[key] = None
            continue
        med[key] = {k: round(statistics.median(r[k] for r in rs), 3) for k in
                    ("prompt_tokens", "gen_tokens", "ttft", "prefill_tps", "decode_tps", "J_per_prompt_token",
                     "J_per_gen_token", "avg_W_decode")}
    code = [r for r in runs if r["prompt"].startswith("code")]
    med["code_all"] = {k: round(statistics.median(r[k] for r in code), 3) for k in
                       ("decode_tps", "J_per_gen_token", "avg_W_decode")}
    res = {"name": a.name, "idle_W": statistics.mean(s[2] for s in idle) / 1000 if idle else None,
           "median": med, "runs": runs}
    (a.out / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    print("MEDIAN", a.name, json.dumps(med), flush=True)


if __name__ == "__main__":
    main()
