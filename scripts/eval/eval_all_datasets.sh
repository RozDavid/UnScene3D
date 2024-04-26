#!/bin/bash

export HYDRA_FULL_ERROR=1

export BASE_PATH="/cluster/himring/drozenberszki/Datasets/Mask3D/data/processed"
export OUTPUT_BASE="./eval_output"
export STD_PARAMS="general.experiment_name=freemask_data_eval general.project_name=scannet general.eval_on_segments=true general.train_on_segments=true general.num_targets=3 data.test_batch_size=1 data.validation_mode=validation data/collation_functions=freemask_voxelize_collate data/datasets=freemask"

# Segment size experiments
#export dataset_1="UnScene3D_DINO_CSC_TAU_0.65"
#export dataset_2="UnScene3D_DINO_TAU_0.65"
#export dataset_3="UnScene3D_CSC_TAU_0.5"
#export dataset_4="unscene3d_dino_csc"
#export dataset_5="unscene3d_dino"
#export dataset_6="scannet_unscene3d"
# export dataset_4="UnScene3D_DINO_CSC_TAU_0.6"
# export dataset_5="UnScene3D_DINO_CSC_TAU_0.65"
# export dataset_6="UnScene3D_DINO_CSC_TAU_0.7"
# export dataset_7="UnScene3D_DINO_CSC_TAU_0.8"
# export dataset_8="UnScene3D_CSC_TAU_0.4"
# export dataset_9="UnScene3D_CSC_TAU_0.5"
# export dataset_10="UnScene3D_CSC_TAU_0.55"
# export dataset_11="UnScene3D_CSC_TAU_0.6"
# export dataset_12="UnScene3D_CSC_TAU_0.65"
# export dataset_13="UnScene3D_CSC_TAU_0.7"
# export dataset_14="UnScene3D_CSC_TAU_0.8"

# Segment size experiments
export dataset_1="UnScene3D_DINO_CSC_SEGMENT_30"
export dataset_2="UnScene3D_DINO_CSC_SEGMENT_50_v2"
export dataset_3="UnScene3D_DINO_CSC_SEGMENT_100"
export dataset_4="UnScene3D_DINO_CSC_SEGMENT_200"
export dataset_5="UnScene3D_DINO_CSC_SEGMENT_400"


# Metric experiments
export dataset_6="UnScene3D_DINO_CSC_SEGMENT_L2_50"
export dataset_7="UnScene3D_DINO_CSC_SEGMENT_L2_100"

# Separation experiments
export dataset_8="UnScene3D_DINO_CSC_NO_SEPARATION"
export dataset_9="UnScene3D_DINO_CSC_LARGEST_SEPARATION"
export dataset_10="UnScene3D_DINO_CSC_AVG_SEPARATION"

# Run the evaluations
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_1}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_1}.txt"
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_2}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_2}.txt"
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_3}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_3}.txt"
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_4}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_4}.txt"
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_5}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_5}.txt"
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_6}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_6}.txt"
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_7}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_7}.txt"
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_8}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_8}.txt"
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_9}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_9}.txt"
python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_10}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_10}.txt"
# python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_11}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_11}.txt"
# python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_12}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_12}.txt"
# python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_13}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_13}.txt"
# python eval_freemask_gt_performance.py $STD_PARAMS "general.data_dir=${BASE_PATH}/${dataset_14}" 2>&1 | tee "${OUTPUT_BASE}/${dataset_14}.txt"


# Eval different modality datasets
#export modal_1="scannet_unscene3d"
#export modal_2="unscene3d_dino"
#export modal_3="unscene3d_dino_csc"
#
#python eval_freemask_gt_performance.py $STD_PARAMS "general.experiment_name=cutler_data_eval_${modal_1}" "general.data_dir=${BASE_PATH}/${modal_1}" 2>&1 | tee "${OUTPUT_BASE}/${modal_1}.txt"
#python eval_freemask_gt_performance.py $STD_PARAMS "general.experiment_name=cutler_data_eval_${modal_2}" "general.data_dir=${BASE_PATH}/${modal_2}" 2>&1 | tee "${OUTPUT_BASE}/${modal_2}.txt"
#python eval_freemask_gt_performance.py $STD_PARAMS "general.experiment_name=cutler_data_eval_${modal_3}" "general.data_dir=${BASE_PATH}/${modal_3}" 2>&1 | tee "${OUTPUT_BASE}/${modal_3}.txt"