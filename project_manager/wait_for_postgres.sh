#!/bin/bash

dockerize -wait tcp://postgresql:5432 -timeout 20s

# Apply database migrations
echo "Apply database migrations"
