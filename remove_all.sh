#!/bin/sh

echo 'tango container down'
docker-compose down

echo '[warning] try to remove all the docker images'
docker rmi $(docker images -q)

echo 'remove labelling/datadb: requires sudo passward'
sudo rm -R labelling/datadb

echo '[warning] try to remove all the volume mounted'
docker volume rm $(docker volume ls -q)

echo '[warning] try to clear all the build caches'
echo 'if you start build all over again, it will take a long time.'
echo '------estimated build time------'
echo 'project_manager:  1m23s'
echo 'labelling:       16m38s'
echo 'autonn:           4m26s'
echo 'bms:              1m43s'
echo 'autonn_yoloe:     1m27s'
echo 'autonn_resnet:      32s'
echo 'code_gen:        19m37s'
echo 'cloud_deploy:       49s'
echo 'kube_deploy:        54s'
echo '--------------------------------'
docker system df
docker builder prune

