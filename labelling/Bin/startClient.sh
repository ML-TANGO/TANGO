#!/bin/bash
source activate BluAI
conda info

cd "/data/Bluai/Client/"
npm run start:prod 1>/data/Bluai/Client.log 2>&1 &