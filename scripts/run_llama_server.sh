#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

exec ./llama.cpp/build/bin/llama-server \
  --model ./models/bge-small-en-v1.5-q8_0.gguf \
  --embedding \
  --host 127.0.0.1 \
  --port 8080 \
  --parallel 8 \
  "$@"
