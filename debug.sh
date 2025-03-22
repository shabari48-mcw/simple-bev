#!/bin/bash

DATA_DIR="/media/ava/Data_CI/Datasets/nuscenes-mini/"
# DATA_DIR="/media/ava/Data_CI/Datasets/nuscenes-full/nuscenes/"

MODEL_NAME="8x5_5e-4_rgb12_22:43:46"

EXP_NAME="00" # evaluate rgb00 model

python debug.py \
       --batch_size=1 \
       --exp_name=${EXP_NAME} \
       --dset='mini' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_nuscenes' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=1 \
       --device_ids=[0]

 