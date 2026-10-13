#!/bin/bash
# Launch check: start the Forge server with its real command, /health, two chat requests, stop. Two times.
#   scripts/m6bench/launch_check.sh bonsai|ornith      (FORGE_MODELS_DIR, M6BENCH_HOST, M6BENCH_ANE_PORT from env)
set -u
FORGE=$(cd "$(dirname "$0")/../.." && pwd)
cd "$FORGE"
H=${M6BENCH_HOST:-127.0.0.1}; P=${M6BENCH_ANE_PORT:-8765}; M=${FORGE_MODELS_DIR:-$HOME/Models}
LOGS=${LOGS:-$FORGE/../m6bench-out}; mkdir -p "$LOGS"
if [ "$1" = bonsai ]; then
  B=$M/forge-bonsai2-taps
  CMD=(env FORGE_FOREIGN_DRAFTER=1 .venv/bin/python forge.py serve --runtime coreai --model $B/model --build $B/coreai
       --draft $B/drafter/dflash2_lut4_gptq.aimodel --drafter $B/drafter --ctx 8192 --host $H --port $P)
else
  B=$M/forge-ornith15
  CMD=(.venv/bin/python forge.py serve --runtime coreai --plain --model $B/model --build $B/coreai --ctx 8192 --host $H --port $P)
fi
for run in 1 2; do
  pkill -f qwen38_server; pkill -f llama-server; sleep 3
  echo "=== $1 launch $run: ${CMD[*]}"
  "${CMD[@]}" > "$LOGS/launch-$1-$run.log" 2>&1 &
  for i in $(seq 1 600); do curl -s -m 3 http://$H:$P/health >/dev/null && break; sleep 3; done
  curl -s http://$H:$P/health | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print("health:", {k: d.get(k) for k in ("status","soc_class","soc_generation","ane_bonded_compile_mode","context","kv_cache_dtype")})'
  for q in "What is the capital of France? Answer in one sentence." "What is 17 times 23? Reply with the number only."; do
    curl -s http://$H:$P/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":40,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
      | .venv/bin/python -c 'import json,sys; r=json.load(sys.stdin); print("answer:", repr(r["choices"][0]["message"]["content"]))'
  done
  pkill -f qwen38_server; sleep 3
done
