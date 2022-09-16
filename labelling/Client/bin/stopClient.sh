#!/bin/sh

PIDS=`ps -ef | grep Client | grep -v shell | grep hktire | grep -v grep | awk '{print $2}'`

for PID in $PIDS
do
	echo $PID
	kill $PID
done