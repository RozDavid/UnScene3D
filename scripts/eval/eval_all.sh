# Init all the experiment names to put the results
#eval_0="general.experiment_name=freemask_CSC_val general.checkpoint=true"
#eval_1='general.experiment_name=unscene3d_CSC_droploss_0.01 general.checkpoint=true'
#eval_2='general.experiment_name=freemask_dino_droploss_0.01 general.checkpoint=true'
#eval_3='general.experiment_name=freemask_lseg_droploss_0.1 general.checkpoint=true'

# Run all self-train experiments
# eval_0="general.experiment_name=unscene3d_DINO_self_train_2 general.checkpoint=true"
# eval_1='general.experiment_name=unscene3d_DINO_CSC_self_train_2 general.checkpoint=true'
#eval_2='general.experiment_name=unscene3d_CSC_self_train_3 general.checkpoint=true'
#eval_3='general.experiment_name=freemask_lseg_self_train_1 general.checkpoint=true'

# Cycle comparisons CSC self trains
# save_path=/cluster/himring/drozenberszki/Datasets/Mask3D/data/self_train/
# eval_0='general.experiment_name=eval_unscene3d_CSC_self_train_1 general.checkpoint="/cluster/himring/drozenberszki/Datasets/Mask3D/data/self_train/unscene3d_CSC_droploss_0.01/epoch=399-val_mean_ap_50=0.180.ckpt"'
# eval_1='general.experiment_name=eval_unscene3d_CSC_self_train_2 general.checkpoint="/cluster/himring/drozenberszki/Datasets/Mask3D/data/self_train/unscene3d_CSC_self_train_1/epoch=99-val_mean_ap_50=0.241.ckpt"'
# eval_2='general.experiment_name=eval_unscene3d_CSC_self_train_3 general.checkpoint="/cluster/himring/drozenberszki/Datasets/Mask3D/data/self_train/unscene3d_CSC_self_train_2/epoch=79-val_mean_ap_50=0.258.ckpt"'
# eval_3='general.experiment_name=eval_unscene3d_CSC_self_train_4 general.checkpoint="/cluster/himring/drozenberszki/Datasets/Mask3D/data/self_train/unscene3d_CSC_self_train_3/epoch=139-val_mean_ap_50=0.265.ckpt"'

# eval_4='general.experiment_name=eval_unscene3d_DINO_CSC_self_train_1 general.checkpoint="/cluster/himring/drozenberszki/Datasets/Mask3D/data/self_train/unscene3d_DINO_CSC_droploss_0.1/epoch=464-val_mean_ap_50=0.234.ckpt"'
# eval_5='general.experiment_name=eval_unscene3d_DINO_CSC_self_train_2 general.checkpoint="/cluster/himring/drozenberszki/Datasets/Mask3D/data/self_train/unscene3d_CSC_DINO_self_train/epoch=114-val_mean_ap_50=0.298.ckpt"'
# eval_6='general.experiment_name=eval_unscene3d_DINO_CSC_self_train_3 general.checkpoint="/cluster/himring/drozenberszki/Datasets/Mask3D/data/self_train/unscene3d_DINO_CSC_self_train_2/epoch=19-val_mean_ap_50=0.319.ckpt"'
# eval_7='general.experiment_name=eval_unscene3d_DINO_CSC_self_train_4 general.checkpoint="/cluster/himring/drozenberszki/Datasets/Mask3D/data/self_train/unscene3d_DINO_CSC_self_train_3/epoch=104-val_mean_ap_50=0.321.ckpt"'

# ArKit self train experiment
eval_0="general.experiment_name=unscene3d_arkit_scannet_ckpt_self_train_2 general.checkpoint=/rhome/drozenberszki/projects/Mask3D-fork/saved/unscene3d_arkit_scannet_ckpt_self_train_2/last.ckpt"

# Necessary params for standard evaluation
export EVAL_PARAMS="general.project_name=mask3d general.train_mode=false general.eval_on_segments=true data.test_batch_size=1 general.num_targets=3 data/datasets=freemask data/collation_functions=freemask_voxelize_collate logging=offline"
export DATA_PARAMS="data.test_dataset.data_dir=data/processed/unscene3d_arkit data.validation_dataset.data_dir=data/processed/unscene3d_arkit"  # scannet_freemask_oracle, unscene3d_arkit

# Parameters if we want to export for self train
export PHASE="data.test_dataset.mode=train_validation"  # train, validation, test, train_validation
export FREEMASK_PARAMS="general.filter_out_instances=true general.save_visualizations=false general.save_for_freemask=true"

# Run everything, let's gooo
python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_0}
# mv saved/eval_unscene3d_CSC_self_train_1/visualizations "${save_path}/unscene3d_CSC_droploss_0.01"
# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_1}
# mv saved/eval_unscene3d_CSC_self_train_2/visualizations "${save_path}/unscene3d_CSC_self_train_1"
# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_2}
# mv saved/eval_unscene3d_CSC_self_train_3/visualizations "${save_path}/unscene3d_CSC_self_train_2"
# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_3}
# mv saved/eval_unscene3d_CSC_self_train_4/visualizations "${save_path}/unscene3d_CSC_self_train_3"

# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_4}
# mv saved/eval_unscene3d_DINO_CSC_self_train_1/visualizations "${save_path}/unscene3d_DINO_CSC_droploss_0.1"
# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_5}
# mv saved/eval_unscene3d_DINO_CSC_self_train_2/visualizations "${save_path}/unscene3d_CSC_DINO_self_train"
# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_6}
# mv saved/eval_unscene3d_DINO_CSC_self_train_3/visualizations "${save_path}/unscene3d_CSC_DINO_self_train_2"
# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_7}
# mv saved/eval_unscene3d_DINO_CSC_self_train_4/visualizations "${save_path}/unscene3d_CSC_DINO_self_train_3"
