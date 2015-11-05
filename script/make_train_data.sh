#! /bin/bash

VOICE_DIR="data/voice"
TRAIN_DIR="data/train"
SCRIPT_DIR="script"
VOICE_LIST="$TRAIN_DIR/voice_list"
MGCEP_ORDER=40

echo 'Making train data...'

rm -rf $TRAIN_DIR/*
speaker_index=0
for dir in $VOICE_DIR/*; do
  speaker=${dir##*/}
  echo "Processing speaker: ${speaker}"
  if [ ! -d ${dir} ]; then
    continue
  fi
  for path in ${dir}/*.mp3; do
    file=${path##*/}
    echo $TRAIN_DIR/${file} >> $VOICE_LIST
    lame --resample 16 -b 32 -a ${path} $TRAIN_DIR/temp.mp3
    lame --decode -t $TRAIN_DIR/temp.mp3 $TRAIN_DIR/temp.raw
    python $SCRIPT_DIR/trim_zero.py $TRAIN_DIR/temp.raw $TRAIN_DIR/temp2.raw
    x2x +sf < $TRAIN_DIR/temp2.raw | pitch -a 1 -p 80 -s 16 -L 60 -H 400 > $TRAIN_DIR/${file}.pitch
    x2x +sf < $TRAIN_DIR/temp2.raw | frame -l 400 -p 80 | window -l 400 -L 512 | mgcep -m $MGCEP_ORDER -a 0.42 -g -0.5 -l 512 -e 1 > $TRAIN_DIR/${file}.mgcep
    rm $TRAIN_DIR/temp.mp3
    rm $TRAIN_DIR/temp.raw
    rm $TRAIN_DIR/temp2.raw
  done
  python $SCRIPT_DIR/concat_pitch_mgcep.py -l $VOICE_LIST -o $TRAIN_DIR/${speaker_index}.pkl -m $MGCEP_ORDER
  rm $TRAIN_DIR/*.pitch
  rm $TRAIN_DIR/*.mgcep
  rm $VOICE_LIST
  echo ${speaker} >> $TRAIN_DIR/speaker.txt
  speaker_index=$((speaker_index + 1))
done

echo 'Completed'
