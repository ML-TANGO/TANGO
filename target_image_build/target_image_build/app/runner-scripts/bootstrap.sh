#!/bin/bash
BAI_MULTINODE_CONFIG_TF=$(/opt/backend.ai/bin/python /opt/container/setup_multinode.py)
if [ -z "$BAI_MULTINODE_CONFIG_TF" ];
then
	echo "";
else
	echo ${BAI_MULTINODE_CONFIG_TF}
	export TF_CONFIG="${BAI_MULTINODE_CONFIG_TF}"
fi
