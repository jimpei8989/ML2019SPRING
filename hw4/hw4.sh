trainCSV=${1}
outputDir=${2}
modelH5="model.h5"
modelMD5=$(cat model.md5 | cut -d ' ' -f 1)

# Download Model on Google Drive
# Reference: https://www.matthuisman.nz/2019/01/download-google-drive-files-wget-curl.html
fileID="1B68je6y0YKDlxku96FAsCAKlHTGgLl9e"

wget --save-cookies cookies.txt "https://docs.google.com/uc?export=download&id="${fileID} -O- \
         | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O ${modelH5} \
         'https://docs.google.com/uc?export=download&id='${fileID}'&confirm='$(<confirm.txt)

rm confirm.txt cookies.txt

# Check file integrity with MD5 Sum
fileMD5=$(md5sum ${modelH5} | cut -d ' ' -f 1)

if [[ ${modelMD5} != ${fileMD5} ]]; then
    echo -e "Warning:\tMD5 sum diff"
else
    echo -e "MD5 sum check: OK!"
fi

# Saliency Map
echo -e "-> SaliencyMap"
time python3 Prob1/SaliencyMap.py ${modelH5} ${trainCSV} ${outputDir}

# Filter Visualization
echo -e "-> Filter Visualization"
time python3 Prob2/FilterVisualize.py ${modelH5} ${outputDir}

# Output Visualization
echo -e "-> Output Visualization"
time python3 Prob2/OutputVisualize.py ${modelH5} ${trainCSV} ${outputDir}

# Lime
echo -e "-> Lime"
time python3 Prob3/Lime.py ${modelH5} ${trainCSV} ${outputDir}


