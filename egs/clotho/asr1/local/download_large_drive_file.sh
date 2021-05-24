#!/bin/bash

fileid=$1         # "1-QO73Elfdhtx8Jo918eJlksZKvTAoidx"
filename=$2       # "tmp/audiocaps_data_train.tar.gz"
cookie_path=./tmp/cookie

curl -c ${cookie_path} -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ${cookie_path} "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ${cookie_path}`&id=${fileid}" -o ${filename}

rm ${cookie_path}