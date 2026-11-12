#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

cmake -S . -B build
cmake --build build --parallel
