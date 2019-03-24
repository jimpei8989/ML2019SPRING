#! /usr/bin/env bash

TrainCSV=${1}
TestCSV=${2}
XTrainCSV=${3}
YTrainCSV=${4}
XTestCSV=${5}
PredictCSV=${6}

TrainPY="Reproduce/Strong2/train.py"
TestPY="Reproduce/Strong2/test.py"
dataNPY="Reproduce/Strong2/data.npz"
ModelPKL="Reproduce/Strong2/model.pkl"

# # Reproduce Training Step
# python3 ${TrainPY} ${XTrainCSV} ${YTrainCSV} ${dataNPY} ${ModelPKL}

# Predict Step
python3 ${TestPY} ${XTestCSV} ${PredictCSV} ${dataNPY} ${ModelPKL}

