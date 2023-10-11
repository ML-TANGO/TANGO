#!/bin/bash
set -e

docker build -f Dockerfile.train -t ghcr.io/tango/pytorch:21.08-py3 .
