#!/bin/bash
source activate BluAI
conda info

# cd "/data/Bluai/Model/Manager/"
nohup python /data/Bluai/Model/Manager/Master.py > Master.log 2>&1 &