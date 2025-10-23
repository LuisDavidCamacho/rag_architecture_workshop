#!/usr/bin/env sh
set -eu

MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
HOST="${OLLAMA_HOST:-0.0.0.0}"
PORT="${OLLAMA_PORT:-11434}"

export OLLAMA_HOST="${HOST}"
export OLLAMA_PORT="${PORT}"

warm_up() {
  echo "Starting temporary Ollama server to warm up model ${MODEL}"
  ollama serve >/tmp/ollama-warmup.log 2>&1 &
  SERVER_PID=$!

  READY=0
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:${PORT}/api/version" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 1
  done

  if [ "$READY" -eq 1 ]; then
    if ! ollama pull "${MODEL}" >/tmp/ollama-pull.log 2>&1; then
      echo "Warning: failed to pull model ${MODEL}. Review /tmp/ollama-pull.log"
    fi
  else
    echo "Warning: Ollama warmup server did not become ready. Skipping pre-pull."
  fi

  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" 2>/dev/null || true
}

warm_up

echo "Launching Ollama server on ${HOST}:${PORT}"
exec ollama serve
