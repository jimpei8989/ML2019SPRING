#! /usr/bin/env bash

TrainCSV=${1}
TestCSV=${2}
XTrainCSV=${3}
YTrainCSV=${4}
XTestCSV=${5}
PredictCSV=${6}

TrainPY="Reproduce/GenerativeModel/train.py"
TestPY="Reproduce/GenerativeModel/test.py"
dataNPY="Reproduce/GenerativeModel/data.npz"

# # Reproduce Training Step
# python3 ${TrainPY} ${XTrainCSV} ${YTrainCSV} ${dataNPY}

# Predict Step
python3 ${TestPY} ${XTestCSV} ${PredictCSV} ${dataNPY}

