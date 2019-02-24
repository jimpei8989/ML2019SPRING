#!/usr/bin/env bash

if [[ $# -ne 2 ]]; then
    echo "Usage: sh cleanup.sh INPUT_FILE OUTPUT_FILE" >&2
    exit
fi

INPUT=$1
OUTPUT=$2

cat ${INPUT} | sed 's/NR/0/g' > ${OUTPUT}
