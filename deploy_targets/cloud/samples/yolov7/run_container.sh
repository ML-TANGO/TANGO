#!/bin/bash
set -e

docker run --rm -it \
    -v $(pwd):/playground \
    nvcr.io/nvidia/pytorch:21.08-py3
