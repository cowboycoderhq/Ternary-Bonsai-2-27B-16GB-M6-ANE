# Second 16 GB M6 Mac mini

The same benchmark (`scripts/m6bench/gpu_matrix.sh` with `cmp_bench.py`) on a second 16 GB M6 Mac mini with the same
macOS version, the same engines, and the same models. Each engine ran alone. Median of 3 runs, 256 generated tokens,
greedy. Energy is whole-machine power from `macmon`.

| Engine | Prefill, 2,048 tok (tok/s) | Decode, code (tok/s) | J per generated token (code) | Watts during decode (code) |
|---|---|---|---|---|
| GPU, llama.cpp (Prism fork) | 70 | 17.2 | 2.21 | 38.0 |
| GPU, mlx-serve 26.10.1 | 183 | 17.1 | 1.29 | 22.1 |
| **ANE, this repo (DFlash2 + prefix cache)** | **303** | **52.6** | **0.24** | **12.9** |

The speeds are within 2 tok/s of the first mini ([m6-gpu-compare](../m6-gpu-compare)). Power is 1 to 3 W higher.

mlx-serve with the DFlash2 drafter did not run on this mini. Its admission check refused the first 512-token prompt
(needed 1824 MB, 1790 MB available). The 36.3 tok/s figure for that engine comes from the first mini only.
