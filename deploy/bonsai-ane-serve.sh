#!/bin/bash
# Ternary Bonsai 2 27B on the M6 Neural Engine: Core AI build + DFlash2 drafter + automatic prefix cache.
# Environment (defaults): FORGE_DIR (~/src/Ternary-Bonsai-2-27B-16GB-M6-ANE/forge, made by scripts/prepare.sh), BUNDLE (~/Models/Ternary-Bonsai-2-27B-M6-ANE), HOST (127.0.0.1), PORT (8765),
# CTX (32768), PREFIX_CACHE_MB (900), PREFIX_CACHE_DISK_GB (16).
FORGE_DIR=${FORGE_DIR:-$HOME/src/Ternary-Bonsai-2-27B-16GB-M6-ANE/forge}
B=${BUNDLE:-$HOME/Models/Ternary-Bonsai-2-27B-M6-ANE}
cd "$FORGE_DIR" || exit 1
export FORGE_FOREIGN_DRAFTER=1 PREFIX_CACHE_MB=${PREFIX_CACHE_MB:-900} PREFIX_CACHE_DISK_GB=${PREFIX_CACHE_DISK_GB:-16}
exec .venv/bin/python forge.py serve --runtime coreai --model "$B/model" --build "$B/coreai" \
  --draft "$B/drafter/dflash2_lut4_gptq.aimodel" --drafter "$B/drafter" --ctx "${CTX:-32768}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"
