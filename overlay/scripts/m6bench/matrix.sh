#!/bin/bash
# M6 matrix: each backend/model alone, 3 runs x {512, 2048} prompt tokens, 256 generated, energy via macmon.
#   OUT=<dir> scripts/m6bench/matrix.sh gpu:bonsai ane:bonsai:draft gpu:ornith ane:ornith
# bench.py reads the machine layout from env (M6BENCH_HOST, FORGE_MODELS_DIR, GGUF_DIR, LLAMA_CPP_DIR, MACMON).
HERE=$(cd "$(dirname "$0")" && pwd)
FORGE=$(cd "$HERE/../.." && pwd)
PY=$FORGE/.venv/bin/python
OUT=${OUT:-$FORGE/../m6bench-out}
for spec in "$@"; do
  b=${spec%%:*}; rest=${spec#*:}; m=${rest%%:*}; flag=""
  [ "$rest" != "$m" ] && flag="--${rest#*:}"
  $PY "$HERE/bench.py" --backend $b --model $m $flag --runs 3 --out "$OUT" 2>&1 | grep -E "MEDIAN|placement|Error|Traceback"
done
echo MATRIX_DONE
