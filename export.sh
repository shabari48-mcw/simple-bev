#!/bin/bash

DATA_DIR="/media/ava/Data_CI/Datasets/nuscenes-mini/"
# DATA_DIR="/media/ava/Data_CI/Datasets/nuscenes-full/nuscenes/"

MODEL_NAME="1x5_3e-4_rgb00_20:04:28"

EXP_NAME="00" # evaluate rgb00 model

python onnx_export.py \
       --batch_size=1 \
       --exp_name=${EXP_NAME} \
       --dset='mini' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_nuscenes' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=1 \
       --device_ids=[0]

# python export_fix.py \
#        --batch_size=1 \
#        --exp_name=${EXP_NAME} \
#        --dset='mini' \
#        --data_dir=$DATA_DIR \
#        --log_dir='logs_eval_nuscenes' \
#        --init_dir="checkpoints/${MODEL_NAME}" \
#        --res_scale=1 \
#        --device_ids=[0]
