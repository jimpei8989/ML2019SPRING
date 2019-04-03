testCSV=${1}
predictCSV=${2}
modelH5="Reproduce/Best/model.h5"
modelMD5=$(cat Reproduce/Best/model.md5 | cut -d ' ' -f 1)

echo $modelMD5

# Download Model on Google Drive
# Reference: https://www.matthuisman.nz/2019/01/download-google-drive-files-wget-curl.html

fileID="1naTq4Lnxk7xEqCnMV8awWeAAOaQ9Pktw"
fileName="Reproduce/Best/model.h5"

wget --save-cookies cookies.txt "https://docs.google.com/uc?export=download&id="${fileID} -O- \
         | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O ${fileName} \
         'https://docs.google.com/uc?export=download&id='${fileID}'&confirm='$(<confirm.txt)

rm confirm.txt cookies.txt

# Check MD5 Sum

tmpMD5=$(md5sum ${modelH5} | cut -d ' ' -f 1)
if [[ ${modelMD5} != ${tmpMD5} ]]; then
    echo -e "Warning:\tMD5 sum diff"
else
    echo -e "MD5 sum check: OK!"
fi

python3 Reproduce/Best/test.py ${testCSV} ${predictCSV} ${modelH5}
