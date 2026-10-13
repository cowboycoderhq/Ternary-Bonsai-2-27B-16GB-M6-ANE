# M6 benchmark records

Produced by `scripts/m6bench/matrix.sh` on a 16 GB M6 mini (macOS 27.0.1). See `docs/M6_BONSAI_ORNITH.md`.

- `summary.json`: medians and per-run speed / energy for llama.cpp Metal and the ANE builds (no decoded text).
- `summary.txt`: the same as a table with the pass/fail of each ANE-vs-GPU comparison.
- `macmon-sample.jsonl`: 40 raw `macmon pipe` samples, used by `tests/test_m6_bench.py`.
- `launch-*.txt`: two cold launches per model (`/health` and two chat answers).
- `eval-*.txt`: an 8-task Python coding eval (thinking off, temperature 0); Bonsai 7/8 (GPU 7/8), Ornith 6/8 (GPU 4/8).
