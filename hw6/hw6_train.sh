trainX=$1
trainY=$2
testX=$3
dictTxt=$4

python3 Reproduce/train.py ${dictTxt} \
                           ${trainX} \
                           ${trainY} \
                           ${testX} \
                           Reproduce/predict.csv \
                           Reproduce/word2vec.model \
                           Reproduce/model.h5

