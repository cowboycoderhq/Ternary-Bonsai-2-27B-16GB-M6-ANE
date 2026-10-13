# Ternary-Bonsai-2-27B-16GB-M6-ANE

**Ternary Bonsai 2 27B on a 16 GB M6 Mac, on the Neural Engine (52 tok/s on code, 12 W).** This repository is a layer of scripts and patches on
[ANEMLL Forge](https://github.com/Anemll/anemll-forge) that runs Prism ML's
[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) and
[Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) through Core AI on the ANE, with a DFlash2 speculative
drafter, an automatic prefix cache, and fixes for agent clients such as Grok Build.

It is an independent project. It is not affiliated with or endorsed by ANEMLL, Prism ML, Ornith, z-lab, ProCreations,
the Qwen Team / Alibaba Cloud or Apple. Almost everything that makes it work was built by others: see
[CREDITS.md](CREDITS.md).

## Results on a 16 GB M6 Mac mini

Ternary Bonsai 2 27B, each engine alone on the same machine, the same prompts, median of 3 runs, 256 generated tokens,
greedy. Energy is whole-machine power from `macmon`. Details: [overlay/docs/M6_BONSAI_ORNITH.md](overlay/docs/M6_BONSAI_ORNITH.md)
and [overlay/docs/results/m6-gpu-compare](overlay/docs/results/m6-gpu-compare).

| Engine | Prefill, 2,048 tok (tok/s) | Decode, prose (tok/s) | Decode, code (tok/s) | J per generated token (code) | Watts during decode |
|---|---|---|---|---|---|
| GPU, llama.cpp (Prism fork) | 71 | 17.3 | 17.9 | 1.97 | 35.3 |
| GPU, mlx-serve 26.10.1 | 185 | 16.3 | 16.9 | 1.16 | 19.7 |
| GPU, mlx-serve 26.10.1 + DFlash2 | 183 | 20.2 | 36.3 | 0.80 | 29.8 |
| **ANE, this repo (DFlash2 + prefix cache)** | **301** | **29.4** | **52.5** | **0.22** | **11.8** |

In Grok Build sessions (tool calls, file edits), decode is about 55 tok/s, with 64% of draft tokens accepted.

Demo: [demo/asteroids.html](demo/asteroids.html), an Asteroids game that the model wrote on the M6 through Grok Build.

Quality (WikiText-2 perplexity, tokens 1,024 to 2,048): Prism pack 6.843, ANE build 6.860. Each layer of the ANE build
is within 1.3% of a PyTorch / MLX reference, with 100% top-1 agreement on 64 tokens
([overlay/docs/results/m6-parity](overlay/docs/results/m6-parity)).

## What this repository adds to ANEMLL Forge

- **Bonsai 2 converter** ([scripts/bonsai2_to_forge.py](scripts/bonsai2_to_forge.py)): the Prism MLX pack to a Forge
  export with no requantization. Ternary codes become a 2-bit LUT {-1, 0, +1} with one least-squares scale for each
  row; Prism's Hadamard signs are kept.
- **Models other than the 64-layer Qwen3.8-27B**: the chunk plan follows `num_hidden_layers`, and `TAPS` comes from
  the environment.
- **Attention fix for 16 query / 4 KV heads** (Ornith 9B): the old head reshape had two equal axes, and the M6 ANE
  compiler swapped them.
- **Foreign drafter opt-in** (`FORGE_FOREIGN_DRAFTER=1`): the published DFlash2 drafter drafts for Bonsai.
- **Automatic prefix cache**: snapshots that carry their own KV rows, a RAM tier and an SSD tier, and pinned system
  prompts that the server prefills at startup.
- **Agent fixes**: a tool call cut off at `max_tokens` is dropped with a note (it is not sent as broken JSON), and a
  tool call that fails the same way three times in a row gets a note and one sampled turn.
- **M6 tools**: per-layer ANE parity, a benchmark with energy for any OpenAI-compatible server, and the recorded
  results.

## Layout

| Path | Contents |
|---|---|
| `scripts/config.sh` | The pinned ANEMLL Forge commit (`6c572a8`) |
| `scripts/prepare.sh` | Clones Forge at that commit into `forge/`, applies `patches/` in order, copies `overlay/`, makes the venv |
| `scripts/check.sh` | Runs `prepare.sh`, then the full test suite in the prepared tree |
| `patches/` | Four changes to Forge files, one per topic, in `git format-patch` form |
| `overlay/` | New files copied into the Forge tree: the Bonsai converter, the parity tool, the M6 benchmark tools, tests, results |
| `deploy/` | The server script, a LaunchAgent template and installer, the Grok Build launcher and config |
| `overlay/docs/` | M6 results and the reproduction notes (copied into `forge/docs/`; the tests read them there) |

## Requirements

- An Apple M6 Mac (tested: Mac mini, 16 GB, macOS 27.0.1). M5 machines run Forge at about half the M6 speed.
- macOS 27 with the Core AI framework, Python 3.11, and the Xcode 27 SDK to build the Swift bridge.
- About 10 GB of disk for the bundle, plus about 6 GB for the ANE compile cache.

## Quick start

```sh
git clone https://github.com/cowboycoderhq/Ternary-Bonsai-2-27B-16GB-M6-ANE.git ~/src/Ternary-Bonsai-2-27B-16GB-M6-ANE
cd ~/src/Ternary-Bonsai-2-27B-16GB-M6-ANE
scripts/prepare.sh        # clones ANEMLL Forge at the pinned commit into ./forge, applies patches/, copies overlay/,
                          # makes ./forge/.venv
(cd forge && bash coreai/swift_bridge/build.sh)   # needs the Xcode 27 SDK; or copy a built libcoreai_bridge.dylib

# Model files (Core AI build, tokenizer, embedding, drafter)
forge/.venv/bin/hf download cowboycoderhq/Ternary-Bonsai-2-27B-M6-ANE --local-dir ~/Models/Ternary-Bonsai-2-27B-M6-ANE

# One-time ANE compile (about 6 minutes on M6), then start the server
(cd forge && .venv/bin/python forge.py compile --build ~/Models/Ternary-Bonsai-2-27B-M6-ANE/coreai \
  --draft ~/Models/Ternary-Bonsai-2-27B-M6-ANE/drafter/dflash2_lut4_gptq.aimodel)
bash deploy/install-launchd.sh               # LaunchAgent: starts at login, restarts on failure
curl -s http://127.0.0.1:8765/health
```

The server speaks the OpenAI API at `http://127.0.0.1:8765/v1` (model id `qwen38-27b-ane`).

### Grok Build

Add [deploy/grok-config.toml](deploy/grok-config.toml) to `~/.grok/config.toml`, put
[deploy/grok-bonsai](deploy/grok-bonsai) on your `PATH`, and run `grok-bonsai` in a project folder. The first message of
the first session prefills the Grok system prompt (about 45 s). After that the server keeps it pinned, also across
restarts.

## Build the model yourself

Run these commands after `scripts/prepare.sh`. The conversion needs about 60 GB of free disk and more than 16 GB of RAM (an M5 Max was used).

```sh
# 1. Prism pack -> Forge export (no GPTQ)
PACK=<Ternary-Bonsai-2-27B-mlx-2bit dir> OUT_MODEL=<bundle>/model OUT_EXPORT=<export> \
  forge/.venv/bin/python forge/scripts/bonsai2_to_forge.py
# 2. Core AI packages (32K entries, taps for the DFlash2 drafter)
MODEL=<bundle>/model EXPORT_DIR=<export> EMBED_NPY=<bundle>/model/embed_tokens_fp16.npy TAPS=5,19,33,47,61 \
  SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 QCONV_INT8=0 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 \
  ATT_INT8MM=s8,s8b,sm8,pvf8 KV_KEYS_T=1 OUT=<builds> \
  forge/.venv-convert/bin/python forge/coreai/qwen38_coreai_build.py all --kv-cache-dtype v8 \
  --ctx 8192,16384,32768 --pctx 8192,16384,32768
# 3. Drafter: the drafter/ folder of anemll/anemll-forge-qwen3.8-27B
```

## Limits

- Tested on one machine. Benchmarks used 8K context and 3 runs for each point. A 29K-token prompt fit at 32K context
  on 16 GB, with free memory down to 5% during prefill.
- The drafter drafts with its own Qwen3.8 head, not Bonsai's. On free prose, 27–41% of draft tokens are accepted.
- After a restart, the server prefills its pinned prompts before it accepts requests (about 80 s for three).
- A cached Core AI specialization can fall back to the GPU without an error. Check with
  `forge/coreai/inspect_coreai_cache.py --strict`.

## License

Code: MIT, copyright 2026 Cowboy Coder ([LICENSE](LICENSE)). The patches change ANEMLL Forge, which is MIT, copyright
2026 ANEMLL contributors ([LICENSE.anemll-forge](LICENSE.anemll-forge)); the Forge code is downloaded, not copied here. Model files on Hugging Face keep their upstream licenses (Apache 2.0 for Bonsai and the drafter, MIT for
Ornith); see [NOTICE](NOTICE). Created using Bonsai by Prism ML.
