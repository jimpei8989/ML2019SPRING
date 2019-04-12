trainCSV=${1}
modelH5="Reproduce/Best/model.h5"
historyPkl="Reproduce/Best/history.pkl"

python3 Reproduce/Best/train.py ${trainCSV} ${modelH5} ${historyPkl}
