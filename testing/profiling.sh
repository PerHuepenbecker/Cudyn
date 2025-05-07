#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <program> <filename MatrixMarket"
    exit 1
fi

PROG = "$1"
MATRIX = "$2"

ncu -o "${PROG}_perf" -f --metrics smsp__thread_inst_executed_per_inst_executed.ratio,smsp__thread_inst_executed_per_inst_executed.pct --page raw ./"PROG" "MATRIX"