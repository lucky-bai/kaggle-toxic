#!/bin/bash
set -e

FASTTEXT='../../fastText/fasttext'
TRAIN_OPTIONS='-wordNgrams 2 -epoch 150 -lr 0.05 -thread 32'

echo 'test'
python3 fasttext_preprocess.py test > test.txt
echo 't1'
python3 fasttext_preprocess.py toxic > t1.txt
echo 't2'
python3 fasttext_preprocess.py severe_toxic > t2.txt
echo 't3'
python3 fasttext_preprocess.py obscene > t3.txt
echo 't4'
python3 fasttext_preprocess.py threat > t4.txt
echo 't5'
python3 fasttext_preprocess.py insult > t5.txt
echo 't6'
python3 fasttext_preprocess.py identity_hate > t6.txt

echo 'Training...'
$FASTTEXT supervised $TRAIN_OPTIONS -input t1.txt -output m1
$FASTTEXT supervised $TRAIN_OPTIONS -input t2.txt -output m2
$FASTTEXT supervised $TRAIN_OPTIONS -input t3.txt -output m3
$FASTTEXT supervised $TRAIN_OPTIONS -input t4.txt -output m4
$FASTTEXT supervised $TRAIN_OPTIONS -input t5.txt -output m5
$FASTTEXT supervised $TRAIN_OPTIONS -input t6.txt -output m6

$FASTTEXT predict-prob m1.bin test.txt 1 > o1.txt
$FASTTEXT predict-prob m2.bin test.txt 1 > o2.txt
$FASTTEXT predict-prob m3.bin test.txt 1 > o3.txt
$FASTTEXT predict-prob m4.bin test.txt 1 > o4.txt
$FASTTEXT predict-prob m5.bin test.txt 1 > o5.txt
$FASTTEXT predict-prob m6.bin test.txt 1 > o6.txt

python3 make_submission.py
