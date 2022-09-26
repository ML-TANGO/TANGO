#!/bin/bash
source activate BluAI
conda info

cd "/data/Bluai/Server/"
npm run build
npm run prod 1>/data/Bluai/Server.log 2>&1 &