#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec mpirun --mca osc ^ucx -np 8 ./build/fast_rag
