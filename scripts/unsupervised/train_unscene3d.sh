#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export EXP_NAME=$1
export EXTRA_ARGS=$2

DATA_DIR=data/processed/unscene3d_dino_csc
VAL_DATA_DIR=data/processed/scannet_freemask_oracle

# TRAIN
python main_instance_segmentation.py \
  general.experiment_name=${EXP_NAME} \
  general.project_name="unscene3d" \
  general.eval_on_segments=true \
  general.train_on_segments=true \
  general.num_targets=3 \
  data.batch_size=8 \
  data.test_batch_size=1 \
  data/collation_functions=freemask_voxelize_collate \
  data/datasets=freemask \
  general.data_dir=${DATA_DIR} \
  data.validation_dataset.data_dir=${VAL_DATA_DIR} \
  data.test_dataset.data_dir=${VAL_DATA_DIR} \
  general.resume=True \
  ${EXTRA_ARGS}