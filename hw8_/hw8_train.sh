trainCSV=$1

modelPath="MNet/model.pkl"

python3 MNet/train.py ${trainCSV} ${modelPath}
