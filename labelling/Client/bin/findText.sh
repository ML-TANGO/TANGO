#!/bin/sh

LIST=`find ../src -name "*.js" | grep  Layouts > jsList.txt`
rm -f result.txt
cat jsList.txt |\
while read line
do
	cat $line |\
	while read file
	do
		if [[ "$file" == *"t(\""* ]]
		then
			echo "[${line}] [${file}]" >> result.txt
		fi 
	done
done
