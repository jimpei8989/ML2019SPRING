#! /usr/bin/env bash

# bash hw1.sh INPUT_CSV OUTPUT_CSV

# TRAIN_CSV="data/train.csv"
INPUT_CSV=$1
OUTPUT_CSV=$2

TRAIN_FILE="$(pwd)/Reproduce/Best/train.py"
TEST_FILE="$(pwd)/Reproduce/Best/test.py"
NP_FILE="$(pwd)/Reproduce/Best/result.npz"

# # Train
# python3 ${TRAIN_FILE} ${TRAIN_CSV} ${NP_FILE}

# Public Score:     5.53925
# Private Score:    6.60050

# Test (Generate output)
python3 ${TEST_FILE} ${NP_FILE} ${INPUT_CSV} ${OUTPUT_CSV}


