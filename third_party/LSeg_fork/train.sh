#!/bin/bash
#python -u train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 4 --exp_name lseg_ade20k_l16 \
#--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384

export BATCH_SIZE=$1
export POSTFIX=$2
export DATA_ROOT="/cluster/himring/drozenberszki/Datasets/ScanNet2D"

python -u train_lseg.py \
--dataset ScanNet2DSegmentationDataset \
--data_path $DATA_ROOT \
--batch_size $BATCH_SIZE \
--exp_name lseg_scannet200_vitb32_$POSTFIX \
--base_lr 0.004 \
--weight_decay 1e-4 \
--no-scaleinv \
--max_epochs 50 \
--widehead \
--augment \
--accumulate_grad_batches 1 \
--backbone clip_vitb32_384 \

#--no_resume
