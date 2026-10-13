# Credits

This repository is a layer of scripts and patches. Almost everything that makes it work was built by others.

## Runtime

- **[ANEMLL Forge](https://github.com/Anemll/anemll-forge)** by the ANEMLL contributors, MIT License. It does the Core
  AI conversion, the ANE compile, the Swift bridge, the DFlash2 speculative decoding and the OpenAI-compatible server.
  Every file in `patches/` is a modification of ANEMLL Forge at commit `6c572a8` (see
  [LICENSE.anemll-forge](LICENSE.anemll-forge)). `scripts/prepare.sh` downloads Forge; its code is not copied here.

## Models

- **[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)** by Prism ML, Apache 2.0.
  Created using Bonsai by Prism ML. Bonsai is built from
  **[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** by the Qwen Team / Alibaba Cloud, Apache 2.0.
- **DFlash2 drafter** from [anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B),
  derived from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2)
  and [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), Apache 2.0.
- **[Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B)** by Ornith AI, MIT License.

The converted model files are on Hugging Face
([cowboycoderhq/Ternary-Bonsai-2-27B-M6-ANE](https://huggingface.co/cowboycoderhq/Ternary-Bonsai-2-27B-M6-ANE),
[cowboycoderhq/Ornith-1.5-9B-LUT4-M6-ANE](https://huggingface.co/cowboycoderhq/Ornith-1.5-9B-LUT4-M6-ANE)), each with its
upstream license, notice and a list of changes.

## Tools used for the comparisons

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (Prism ML fork for Bonsai), MIT.
- [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu, MIT.
- [macmon](https://github.com/vladkens/macmon) by vladkens, MIT, for the energy measurements.
- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart (tested; version 1.0.8 does not run Bonsai).

## Independence

This project is not affiliated with or endorsed by ANEMLL, Prism ML, Ornith, z-lab, ProCreations, the Qwen Team /
Alibaba Cloud, Apple, or the authors of the tools above.
