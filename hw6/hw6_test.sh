testX=$1
dictTxt=$2
predictY=$3

w2vModel="Reproduce/word2vec.model"
modelH5="Reproduce/model.h5"

# Download Model on Google Drive
# Reference: https://www.matthuisman.nz/2019/01/download-google-drive-files-wget-curl.html
## W2V model
w2vID="1ewCBdiGRV__24GL1SDmJKvm4Y9WEe7N-"

wget --save-cookies cookies.txt "https://docs.google.com/uc?export=download&id="${w2vID} -O- \
             | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O ${w2vH5} \
             'https://docs.google.com/uc?export=download&id='${w2vID}'&confirm='$(<confirm.txt)

## RNN model
modelID="1rSsWay_EVChFjttf5F3jcKnQn0po-L0g"

wget --save-cookies cookies.txt "https://docs.google.com/uc?export=download&id="${modelID} -O- \
             | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O ${modelH5} \
             'https://docs.google.com/uc?export=download&id='${modelID}'&confirm='$(<confirm.txt)

rm confirm.txt cookies.txt


python3 Reproduce/test.py ${dictTxt} \
                          ${testX} \
                          ${predictY} \
                          ${w2vModel} \
                          ${modelH5}

