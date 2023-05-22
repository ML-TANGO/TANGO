#!/bin/bash

# initial file
# FILE="/var/app/Server/dist/config/initial_date.txt"
DBFILE="/var/app/bin/init.sh"

# if [ -f ${FILE} ]
# then
# 	DATE=`cat ${FILE}`
# 	echo "BluAI initialized at ${DATE}"
# else
# 	DATE=`date '+%Y%m%d%H%M%S'`
# 	echo ${DATE} > ${FILE}
# fi

# database start
if [ -f ${DBFILE} ]
then
  ${DBFILE} mysqld
	# rm -f ${DBFILE}
	# rm -rf /docker-entrypoint-initdb.d
else
	./database.sh mysqld
fi


LOADED=true
while $LOADED
do
	echo "check process..."
	PROCESS=`ps -ef | grep mysqld | grep -v grep | awk '{print $8}'`
	if [ "${PROCESS}" == "mysqld" ]
	then
		LOADED=false
	fi
	sleep 2
done

# model start
cd ../Model
nohup python Manager/Master.py > /dev/null 2>&1 &

# nginx start
service nginx start

# server start
cd ../Server
NODE_ENV=production node ./dist/index.js

