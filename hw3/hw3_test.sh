testCSV=${1}
predictCSV=${2}
modelH5="Reproduce/Best/model.h5"
modelChecksum=$(cat Reproduce/Best/model.md5)

python3 Reproduce/Best/test.py ${testCSV} ${predictCSV} ${modelH5}


