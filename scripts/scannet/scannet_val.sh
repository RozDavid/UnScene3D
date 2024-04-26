#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export EXP_NAME=$1
export EXTRA_ARGS=$2

# TRAIN
python main_instance_segmentation.py \
general.experiment_name=${EXP_NAME} \
general.eval_on_segments=true \
general.train_on_segments=true \
general.project_name="mask3d" \
general.resume=True \
${EXTRA_ARGS}


CURR_DBSCAN=0.95
CURR_TOPK=500
CURR_QUERY=150

# TEST
#python main_instance_segmentation.py \
#general.experiment_name="${EXP_NAME}_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}" \
#general.project_name="mask3d" \
#general.checkpoint="saved/${EXP_NAME}/last-epoch.ckpt" \
#general.train_mode=false \
#general.eval_on_segments=true \
#general.train_on_segments=true \
#model.num_queries=${CURR_QUERY} \
#general.topk_per_image=${CURR_TOPK} \
#general.use_dbscan=true \
#general.dbscan_eps=${CURR_DBSCAN}
