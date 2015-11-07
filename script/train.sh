#! /bin/bash

GPU_ID="-1"

while getopts g: OPT
do
  case $OPT in
    "g" ) GPU_ID="$OPTARG" ;;
  esac
done

python script/train.py -g $GPU_ID -o data/model/train.pkl -t data/train --iter 200
