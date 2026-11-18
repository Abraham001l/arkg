#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if (( $# < 1 )); then
    echo "Usage: $0 [MPI options] <file_host_url>" >&2
    exit 1
fi

mpi_args=("${@:1:$#-1}")
file_host_url=${!#}
exec mpirun --mca osc '^ucx' -np 8 "${mpi_args[@]}" ./build/arkg_builder "$file_host_url"
