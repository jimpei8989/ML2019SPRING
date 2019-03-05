#! /usr/bin/env bash

# bash hw1.sh INPUT_CSV OUTPUT_CSV

INPUT_CSV=$1
OUTPUT_CSV=$2

TRAIN_FILE="$(pwd)/Release/Simple/train.py"
TEST_FILE="$(pwd)/Release/Simple/test.py"
NP_FILE="$(pwd)/Release/Simple/result.npz"

# # Train
# python3 ${TRAIN_FILE} ${TRAIN_CSV} ${NP_FILE}

# Test (Generate output)
python3 ${TEST_FILE} ${NP_FILE} ${INPUT_CSV} ${OUTPUT_CSV}

