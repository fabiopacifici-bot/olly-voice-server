#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"/..
# Simple health-check loop before sending the POST. Exits 1 if server not ready.
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8766}
URL="http://${HOST}:${PORT}/health"
TRIES=5
SLEEP=1
ready=0
for i in $(seq 1 ${TRIES}); do
  if curl -sSf "${URL}" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep ${SLEEP}
done
if [ "${ready}" -ne 1 ]; then
  echo "SERVER NOT RUNNING: could not reach ${HOST}:${PORT} after ${TRIES} tries"
  exit 1
fi

# Ensure server running separately
TEXT="Hello from smoke test"
OUT=assets/smoke_out.wav
curl -s -X POST -F "text=${TEXT}" http://${HOST}:${PORT}/tts -o ${OUT}
if [ -s ${OUT} ]; then
  echo "OK: output saved to ${OUT}"
  exit 0
else
  echo "FAIL: no output"
  exit 2
fi
