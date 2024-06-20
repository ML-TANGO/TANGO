#!/bin/sh

docker-compose down
docker rmi $(docker images -q)
sudo rm -R labelling/datadb
docker-compose up --build
