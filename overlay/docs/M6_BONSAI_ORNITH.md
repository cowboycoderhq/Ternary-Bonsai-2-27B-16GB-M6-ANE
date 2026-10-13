# Ternary Bonsai 2 27B and Ornith-1.5-9B on the M6 ANE

Two more Qwen3.5-family checkpoints through the Core AI pipeline, measured on a 16 GB M6 mini against llama.cpp
Metal on the same machine. Measurements, not projections; see the limits at the end.

> Paths in this document are inside the Forge tree that `scripts/prepare.sh` makes (`forge/` by default): changed Forge files come from `patches/`, new files from `overlay/`.

## What changed

| Change | Files | Why |
|---|---|---|
| Bonsai 2 converter | `scripts/bonsai2_to_forge.py` | Reads the `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` pack (ternary codes, 128-group scales, Walsh-Hadamard with explicit signs) and writes a Forge MODEL + export dir without GPTQ: the codes become a 2-bit LUT {-1, 0, +1} with one least-squares scale per output row; the embedding is dequantized and un-rotated once. |
| Explicit sign vectors | `coreai/qwen38_coreai_build.py` (`explicit_signs`, `Hadamard`, `Head`) | A rotation seed found in `MODEL/bonsai_signs.npz` uses the stored sign vector instead of the seeded RNG; `lm_head.safetensors` may carry `rot_seed`, rotating the head input. Builds without the file are unchanged. |
| Layer-count-driven plans | `forge.py`, `coreai/qwen38_coreai_build.py`, `scripts/qwen38_ane_model.py` | The 64-layer / 5120-hidden check becomes "layer count divisible by 4, `full_attention_interval` 4"; the chunk plan follows `num_hidden_layers`; `TAPS` can be set from the environment (empty for no drafter). |
| Attention head layout | `coreai/qwen38_coreai_build.py` (`AttnW.forward`) | Query rows per KV head are formed head-major (`permute(1, 0, 2).reshape(nkv, grp*T, hd)`), and the output is gathered the same way. Mathematically identical to the old form; but for grp == nkv (Ornith 9B: 16 query / 4 KV heads) the old `reshape(T, nkv, grp, hd).permute(1, 2, 0, 3)` has two equal-size axes and the M6 ANE lowering swapped them (per-head error 3% for heads 0, 5, 10, 15 and 20-65% for the rest). Unchanged results for the 27B (grp 6). |
| Foreign drafter opt-in | `scripts/hf_release.py` | `FORGE_FOREIGN_DRAFTER=1` skips only the target/head export-name check, so the published DFlash2 drafter (which carries its own Qwen3.8 head) can draft for another target with the same taps and shapes. Tap / T / shape / selector checks still apply. |
| Per-layer parity tool | `scripts/coreai_layer_parity.py` | `ref` saves the residual stream after each tapped layer from `dflash2_target_ref.Target` on the same export; `ane` runs the build in T=8 verify calls and reports relative L2 / cosine per layer and logits top-1 agreement. |
| M6 benchmark harness | `scripts/m6bench/` | One backend and model per run, fixed WikiText prompts (512 / 2048 tokens), speed from the servers' timing fields, energy from `macmon` (IOReport rails + SMC system power, no sudo), and a gate that refuses ANE runs whose cached specializations are not `fully_ane`. |

## Results (M6 mini, Mac18,5, 16 GB, macOS 27.0.1, bonded compile mode 2)

Medians of 3 runs per prompt size, 256 generated tokens, thinking off, temperature 0. GPU = llama.cpp Metal
(`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, `Ornith-1.5-9B-Q4_K_M.gguf`, Prism llama.cpp fork). Energy is whole-machine
power over each phase; idle was 1.2-1.3 W. Raw medians and per-run values: `docs/results/m6-bench/summary.json`.

| Model | Prompt | Prefill tok/s GPU -> ANE | Decode tok/s GPU -> ANE | J / prompt token | J / generated token |
|---|---|---|---|---|---|
| Bonsai 2 27B (+ DFlash2) | 512 | 67.7 -> 287.3 | 17.5 -> 39.7 | 0.324 -> 0.040 | 1.92 -> 0.39 |
| Bonsai 2 27B (+ DFlash2) | 2048 | 70.8 -> 313.9 | 17.4 -> 33.1 | 0.346 -> 0.054 | 2.00 -> 0.49 |
| Ornith-1.5-9B (plain) | 512 | 267.6 -> 846.3 | 24.1 -> 25.8 | 0.063 -> 0.015 | 0.94 -> 0.57 |
| Ornith-1.5-9B (plain) | 2048 | 283.6 -> 939.0 | 23.8 -> 25.9 | 0.085 -> 0.015 | 0.99 -> 0.65 |

Bonsai DFlash2 acceptance (greedy, block 8): 27-43% per request, 2.9-4.0 tokens per verify call. Plain Bonsai on the
ANE decodes at 13.7 tok/s, so the drafter is what lifts decode above the GPU.

Quality:

| | Bonsai 2 27B | Ornith-1.5-9B |
|---|---|---|
| WikiText-2 ppl, tokens [1024, 2048), source | 6.843 (Prism MLX pack) | 7.338 (bf16, transformers) |
| Same window, ANE build (`m6_long_ctx_eval.py`) | 6.860 | 7.550 |
| Per-layer parity vs reference (`docs/results/m6-parity`) | < 1.3% rel L2, cos > 0.9999, top-1 100% | same |

Bonsai's per-row scale costs 0.002 ppl over the pack's group scales in MLX (6.845 vs 6.843). The shipped Ornith export
is GPTQ `LUT4 per-tensor + pcs` with activation weighting and 26 in-domain chat rows mixed into the calibration
(layer-0 mixer kept bf16); a 3-bit `vector 2x64` export reached a similar perplexity but noticeably worse code.

## Reproduce

```sh
# Bonsai 2: convert the pack, build with the drafter's taps, compile on the M6, serve with the published drafter
PACK=<pack dir> OUT_MODEL=<bundle>/model OUT_EXPORT=<export> python scripts/bonsai2_to_forge.py
MODEL=<bundle>/model EXPORT_DIR=<export> EMBED_NPY=<bundle>/model/embed_tokens_fp16.npy TAPS=5,19,33,47,61 \
  OUT=<builds> .venv-convert/bin/python coreai/qwen38_coreai_build.py all --kv-cache-dtype v8 --ctx 8192,16384 --pctx 8192,16384
FORGE_FOREIGN_DRAFTER=1 python forge.py serve --runtime coreai --model <bundle>/model --build <build> \
  --draft <drafter>/dflash2_lut4_gptq.aimodel --drafter <drafter> --ctx 8192

# Ornith 1.5 9B: GPTQ with the usual qwen38_gptq_27b.py flow (FORMAT/MIXER/HEAD="LUT4 per-tensor + pcs", AW=1,
# PLAN={"0.mixer": "bf16"}), then the same build with TAPS= (no drafter) and serve --plain.

# Checks
python -m unittest discover -s tests -v
MODEL=<checkpoint> EXPORT_DIR=<export> .venv-convert/bin/python -m unittest tests.test_attn_head_layout tests.test_gdn_fast
scripts/m6bench/matrix.sh gpu:bonsai ane:bonsai:draft gpu:ornith ane:ornith
```

## Limits

- One machine, 8K context (16K / 32K entries built but not benchmarked), 3 runs per point.
- Ornith's decode lead is small (about 8%); it has no drafter yet. `ornith-ai/Ornith-1.5-9B-DFlash` (block 16, 8 taps)
  would need a drafter build path for its shapes.
- The DFlash2 drafter drafts with its own Qwen3.8 head, not Bonsai's; acceptance is correspondingly lower than on its
  own target.
- Energy comes from `macmon` (IOReport + SMC), sampled every ~110 ms; it was not cross-checked against `powermetrics`.
- A cached Core AI specialization fell back to the GPU once (`gpu_regions_present`, 20 s per draft). Purging that
  package's cache and reprobing fixed it; the bench now refuses such runs.
