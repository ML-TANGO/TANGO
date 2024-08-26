#!/bin/sh

echo 'make all docker containers exit and removed'
docker compose down

echo 'remove all docker images'
docker rmi $(docker images -q)

echo 'delete labelling/datadb to avoid permission error'
sudo rm -R labelling/datadb

echo 'build all docker containers'
docker compose build
