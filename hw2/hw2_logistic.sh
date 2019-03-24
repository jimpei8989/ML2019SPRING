#! /usr/bin/env bash

TrainCSV=${1}
TestCSV=${2}
XTrainCSV=${3}
YTrainCSV=${4}
XTestCSV=${5}
PredictCSV=${6}

TrainPY="Reproduce/LogisticRegression/train.py"
TestPY="Reproduce/LogisticRegression/test.py"
dataNPY="Reproduce/LogisticRegression/data.npz"

# # Reproduce Training Step
# python3 ${TrainPY} ${XTrainCSV} ${YTrainCSV} ${dataNPY}

# Predict Step
python3 ${TestPY} ${XTestCSV} ${PredictCSV} ${dataNPY}

