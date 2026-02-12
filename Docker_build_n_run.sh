#!/bin/bash

docker build . -t stfnets
docker run --gpus all -it --rm -v $(pwd):/app stfnets python STFNets.py $1