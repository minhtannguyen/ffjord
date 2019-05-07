#!/bin/bash
DATA_DIR=/tandatasets/small_imnet

mkdir ${DATA_DIR}
cd ${DATA_DIR}

for FILENAME in train_32x32.tar valid_32x32.tar train_64x64.tar valid_64x64.tar
do
    curl -O http://image-net.org/small/$FILENAME
    tar -xvf $FILENAME
done