imgDir=$1
testCsv=$2
predCsv=$3
modelPath="AE.pkl"

python3 cluster.py ${imgDir} ${modelPath} ${testCsv} ${predCsv}
