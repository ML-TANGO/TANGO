#!/bin/sh

docker-compose down
docker rmi $(docker images -q)
sudo rm -R labelling/datadb
docker-compose up --build
echo 'go and try http://localhost:8085 or http://0.0.0.0:8085'
