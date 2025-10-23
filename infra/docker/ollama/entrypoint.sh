#!/usr/bin/env sh
set -eu

MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
HOST_WITH_PORT="${OLLAMA_HOST:-0.0.0.0}"
PORT="${OLLAMA_PORT:-}"

case "${HOST_WITH_PORT}" in
  *:*)
    HOST_PART="${HOST_WITH_PORT%:*}"
    PORT="${PORT:-${HOST_WITH_PORT##*:}}"
    HOST="${HOST_PART}"
    ;;
  *)
    HOST="${HOST_WITH_PORT}"
    PORT="${PORT:-12434}"
    ;;
esac

PORT="${PORT:-12434}"
export OLLAMA_PORT="${PORT}"
export OLLAMA_HOST="${HOST}:${PORT}"

warm_up_model() {
  echo "Starting temporary Ollama server to pre-pull model: ${MODEL}"
  ollama serve >/tmp/ollama-warmup.log 2>&1 &
  SERVER_PID=$!

  READY=0
  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/api/version" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 1
  done

  if [ "$READY" -eq 1 ]; then
    if ! ollama pull "${MODEL}"; then
      echo "Warning: failed to pull model ${MODEL}. The server will continue to start."
    fi
  else
    echo "Warning: temporary Ollama server did not become ready. Skipping model pull."
  fi

  kill "${SERVER_PID}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" 2>/dev/null || true
}

warm_up_model

echo "Launching Ollama server on ${HOST}:${PORT}"
exec ollama serve
