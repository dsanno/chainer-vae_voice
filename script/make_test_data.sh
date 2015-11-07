#! /bin/bash

TEST_DIR="data/test"
SCRIPT_DIR="script"
VOICE_LIST="data/test/voice_list"
MGCEP_ORDER=40

in_path=$1
out_path=$2
file=${in_path##*/}
echo "Making test data... for ${file}"

lame --resample 16 -b 32 -a ${in_path} $TEST_DIR/temp.mp3
lame --decode -t $TEST_DIR/temp.mp3 $TEST_DIR/temp.raw
python $SCRIPT_DIR/trim_zero.py $TEST_DIR/temp.raw $TEST_DIR/temp2.raw
x2x +sf < $TEST_DIR/temp2.raw | pitch -a 1 -p 80 -s 16 -L 60 -H 400 > $TEST_DIR/${file}.pitch
x2x +sf < $TEST_DIR/temp2.raw | frame -l 400 -p 80 | window -l 400 -L 512 | mgcep -m $MGCEP_ORDER -a 0.42 -g -0.5 -l 512 -e 1 > $TEST_DIR/${file}.mgcep
echo $TEST_DIR/${file} > $VOICE_LIST
python $SCRIPT_DIR/concat_pitch_mgcep.py -l $VOICE_LIST -o $2 -m $MGCEP_ORDER
rm $TEST_DIR/temp.mp3
rm $TEST_DIR/temp.raw
rm $TEST_DIR/temp2.raw
rm $TEST_DIR/${file}.pitch
rm $TEST_DIR/${file}.mgcep
rm $VOICE_LIST

echo 'Completed'
