testX=$1
dictTxt=$2
predictY=$3

w2vModel="Reproduce/word2vec.model"
modelH5="Reproduce/model.h5"

wget https://www.csie.ntu.edu.tw/~wjpei/HYML/hw6/word2vec.model -O ${w2vModel}
wget https://www.csie.ntu.edu.tw/~wjpei/HYML/hw6/model.h5 -O ${modelH5}

python3 Reproduce/test.py ${dictTxt} \
                          ${testX} \
                          ${predictY} \
                          ${w2vModel} \
                          ${modelH5}

