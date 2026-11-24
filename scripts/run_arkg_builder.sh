#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if (( $# < 2 )); then
    echo "Usage: $0 [MPI options] <source_host_url> <graph_host_url>" >&2
    exit 1
fi

mpi_args=("${@:1:$#-2}")
source_host_url=${@: -2:1}
graph_host_url=${!#}
exec mpirun --mca osc '^ucx' -np 8 "${mpi_args[@]}" ./build/arkg_builder "$source_host_url" "$graph_host_url"
