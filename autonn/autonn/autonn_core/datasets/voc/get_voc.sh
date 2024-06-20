#!/usr/bin/bash

# Download/unzip
dir = '/shared/datasets/voc'
url = 'http://github.com/ultralytics/yolov5/release/download/v1.0/'
f = 'VOCtrainval_11-May-2012.zip' # 1.95GB, 17126 images
echo 'Downloading' $urls ' ...'
curl -L $url$f -o $f -# && unzip -q $f -d $dir && rm $f &