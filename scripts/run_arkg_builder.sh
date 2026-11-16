#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if (( $# < 2 )); then
    echo "Usage: $0 [MPI options] <file_host_url> <directory>" >&2
    exit 1
fi

mpi_args=("${@:1:$#-2}")
builder_args=("${@: -2}")
exec mpirun --mca osc '^ucx' -np 8 "${mpi_args[@]}" ./build/arkg_builder "${builder_args[@]}"
