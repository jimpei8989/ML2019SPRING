model=$1

python3 ${model}/train.py data/dict.txt.big \
                          data/train_x.csv \
                          data/train_y.csv \
                          data/test_x.csv \
                          ${model}/predict.csv \
                          word2vec.model \
                          ${model}/model.h5

