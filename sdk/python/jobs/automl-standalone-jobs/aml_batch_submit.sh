#!/bin/bash

DIR=/work/irisml/Irisml-Internal/tmp_dir/fulltrain

for d in marsa-regular-ic-benchmark b92-regular-ic-benchmark dtd kitti-distance sco-sg-segmented-batch1-v1
do
  python aml_automl_ic_pipeline.py -d $DIR$d -n $d -dd $d" dataset full train" -tr train -vr val
done