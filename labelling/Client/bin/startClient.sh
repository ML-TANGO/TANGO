#!/bin/sh
CLIENT="/home/hktire/Client"
cd $CLIENT
npm run start:prod 1>$CLIENT/client.log 2>&1 &
