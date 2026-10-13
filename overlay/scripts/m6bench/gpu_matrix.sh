#!/bin/bash
# M6 engine comparison for Ternary Bonsai 2 27B. Runs each engine alone with the same client bench (cmp_bench.py),
# then the ANE server (launchd com.user.bonsai-ane) for the same bench, and starts that server again at the end.
#   scripts/m6bench/gpu_matrix.sh [engine ...]
#   engines: llamacpp mlxserve mlxserve-dflash mlxserve-dflash-always mlxserve-gate-default tensorfold ane
# Paths (override with env): FORGE_DIR, PACK (Prism MLX pack), DFLASH_DIR (z-lab DFlash2 bf16), GGUF (Prism GGUF),
# LLAMA_SERVER, MLX_SERVE, MLX_SERVE_GATE (mlx-serve built with MLX_SERVE_DFLASH_GATE_MIN), TENSORFOLD, OUT, LOGS.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
FORGE_DIR=${FORGE_DIR:-$(cd "$HERE/../.." && pwd)}
OUT=${OUT:-$HOME/m6bench-out/gpucmp}
LOGS=${LOGS:-$HOME/m6bench-out/logs}
mkdir -p "$OUT" "$LOGS"
PY=$FORGE_DIR/.venv/bin/python
BENCH=$HERE/cmp_bench.py
PACK=${PACK:-$HOME/models/bonsai2-mlx}
DFLASH_DIR=${DFLASH_DIR:-$HOME/models/dflash2}
GGUF=${GGUF:-$HOME/models/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf}
LLAMA_SERVER=${LLAMA_SERVER:-$HOME/models/llamacpp/llama-server}
MLX_SERVE=${MLX_SERVE:-mlx-serve}
MLX_SERVE_GATE=${MLX_SERVE_GATE:-$MLX_SERVE}
TENSORFOLD=${TENSORFOLD:-tensorfold-native}
HOST=127.0.0.1
ENGINES=${*:-llamacpp mlxserve mlxserve-dflash ane}

stop_all() {
  launchctl bootout gui/$(id -u)/com.user.bonsai-ane 2>/dev/null
  pkill -f qwen38_server; pkill -f llama-server; pkill -f "mlx-serve serve"; pkill -f "mlx-serve --model"
  pkill -f tensorfold-native; sleep 5
  for i in $(seq 1 30); do  # the next engine needs the port and the memory back
    pgrep -f "qwen38_server|llama-server|mlx-serve|tensorfold-native" >/dev/null || break
    pkill -9 -f "llama-server|mlx-serve|tensorfold-native"; sleep 2
  done
}

wait_up() {  # $1 url, $2 log
  for i in $(seq 1 360); do
    curl -s -m 3 "$1/v1/models" >/dev/null && return 0
    sleep 5
  done
  echo "TIMEOUT starting $1"; tail -20 "$2"; return 1
}

run_engine() {
  local name=$1 port=$2 log=$LOGS/gpucmp-$1.log; shift 2
  stop_all
  echo "=== $name: $*" | tee -a "$OUT/runner.log"
  "$@" > "$log" 2>&1 &
  if wait_up "http://$HOST:$port" "$log"; then
    echo "mem after load: $(memory_pressure | tail -1)" | tee -a "$OUT/runner.log"
    $PY $BENCH --name "$name" --url "http://$HOST:$port" --out "$OUT" --runs 3 --gen 256 2>&1 | grep -E "MEDIAN|Error|Traceback" | tee -a "$OUT/runner.log"
    echo "swap: $(sysctl -n vm.swapusage)" | tee -a "$OUT/runner.log"
  fi
}

for e in $ENGINES; do
  case $e in
    llamacpp) run_engine llamacpp 8080 "$LLAMA_SERVER" -m "$GGUF" \
                --host $HOST --port 8080 -ngl 99 -c 8192 -fa on --jinja -np 1 ;;
    mlxserve) run_engine mlxserve 8091 "$MLX_SERVE" --serve --model "$PACK" --host $HOST --port 8091 --ctx-size 8192 \
                --no-mtp --no-drafter --temp 0 --prefix-cache-entries 0 ;;
    mlxserve-dflash) run_engine mlxserve-dflash 8091 "$MLX_SERVE" --serve --model "$PACK" --host $HOST --port 8091 \
                --ctx-size 8192 --no-mtp --drafter "$DFLASH_DIR" --temp 0 --prefix-cache-entries 0 ;;
    mlxserve-dflash-always) run_engine mlxserve-dflash-always 8091 env MLX_SERVE_DFLASH_GATE_MIN=0 \
                "$MLX_SERVE_GATE" --serve --model "$PACK" --host $HOST --port 8091 \
                --ctx-size 8192 --no-mtp --drafter "$DFLASH_DIR" --temp 0 --prefix-cache-entries 0 ;;
    mlxserve-gate-default) run_engine mlxserve-gate-default 8091 \
                "$MLX_SERVE_GATE" --serve --model "$PACK" --host $HOST --port 8091 \
                --ctx-size 8192 --no-mtp --drafter "$DFLASH_DIR" --temp 0 --prefix-cache-entries 0 ;;
    tensorfold) run_engine tensorfold 8090 "$TENSORFOLD" serve "$PACK" --host $HOST --port 8090 \
                --context 8192 --no-drafts --no-thinking --temperature 0 --parallel 1 --no-update-check ;;
    ane) stop_all
         echo "=== ane: launchd com.user.bonsai-ane (Forge, DFlash2, APC)" | tee -a "$OUT/runner.log"
         launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.bonsai-ane.plist
         if wait_up "http://${ANE_HOST:-127.0.0.1}:${ANE_PORT:-8765}" "$LOGS/bonsai-ane.log"; then
           $PY $BENCH --name ane-dflash --url "http://${ANE_HOST:-127.0.0.1}:${ANE_PORT:-8765}" --out "$OUT" --runs 3 --gen 256 2>&1 | grep -E "MEDIAN|Error|Traceback" | tee -a "$OUT/runner.log"
         fi ;;
  esac
done
stop_all
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.bonsai-ane.plist
echo GPU_MATRIX_DONE | tee -a "$OUT/runner.log"
