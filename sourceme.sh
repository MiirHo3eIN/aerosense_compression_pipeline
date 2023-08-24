#!/bin/bash

# Get the directory the script is in
#DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)

#$ Walk up to root of branch dir
#DIR=$DIR/../../..

PWD=$(pwd)
DIR_CONFIG=$PWD/configs/


export PYTHONPATH=$DIR_CONFIG

echo $PYTHONPATH