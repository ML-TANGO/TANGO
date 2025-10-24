# Arguments (optional) Usage: bash data/scripts/get_imagenet.sh --train --val
if [ "$#" -gt 0 ]; then
  for opt in "$@"; do
    case "${opt}" in
    --train) train=true ;;
    --val) val=true ;;
    esac
  done
else
  train=true
  val=true
fi

# Make dir
d='../datasets/imagenet' # unzip directory
mkdir -p $d && cd $d

# Download/unzip train
if [ "$train" == "true" ]; then
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar # download 138G, 1281167 images
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME; do
    mkdir -p "${NAME%.tar}"
    tar -xf "${NAME}" -C "${NAME%.tar}"
    rm -f "${NAME}"
  done
  cd ..
fi

# Download/unzip val
if [ "$val" == "true" ]; then
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar # download 6.3G, 50000 images
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash # move into subdirs
  rm -r ILSVRC2012_img_val.tar
fi
