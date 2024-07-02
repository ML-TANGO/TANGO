#!/bin/sh

docker system df
docker-compose down
docker rmi $(docker images -q)
docker volume rm $(docker volume ls -q)
docker builder prune


