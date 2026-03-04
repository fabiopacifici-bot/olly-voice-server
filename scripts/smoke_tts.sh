#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Ensure server running separately
TEXT="Hello from smoke test"
OUT=assets/smoke_out.wav
curl -s -X POST -F "text=${TEXT}" http://127.0.0.1:8766/tts -o ${OUT}
if [ -s ${OUT} ]; then
  echo "OK: output saved to ${OUT}"
  exit 0
else
  echo "FAIL: no output"
  exit 2
fi
