testCSV=$1
predictCSV=$2
modelPath="MNet/model.pkl"

python3 MNet/test.py ${modelPath} ${testCSV} ${predictCSV}

