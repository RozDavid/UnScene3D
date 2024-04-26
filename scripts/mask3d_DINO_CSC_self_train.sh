self_train_base_path="data/self_train"
first_experiment_name="DINO_CSC_self_train"
experiments_base_name="DINO_CSC_self_train"

# Second self-train
. scripts/freemask/train_unscene3d.sh "${experiments_base_name}_2" "trainer.max_epochs=70 optimizer=adamw_lower data.train_dataset.self_train_data_dir=${self_train_base_path}/${first_experiment_name} general.checkpoint=${self_train_base_path}/${first_experiment_name} data.train_dataset.load_self_train_data=true"
python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} general.experiment_name="${experiments_base_name}_2" general.checkpoint=true
mv saved/${experiments_base_name}_2 ${self_train_base_path}/${experiments_base_name}_2

# Third self-train
. scripts/freemask/train_unscene3d.sh "${experiments_base_name}_3" "trainer.max_epochs=70 optimizer=adamw_lower data.train_dataset.self_train_data_dir=${self_train_base_path}/${experiments_base_name}_2 general.checkpoint=${self_train_base_path}/${experiments_base_name}_2 data.train_dataset.load_self_train_data=true"
python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} general.experiment_name="${experiments_base_name}_3" general.checkpoint=true
mv saved/${experiments_base_name}_3 ${self_train_base_path}/${experiments_base_name}_3

# Forth self-train
. scripts/freemask/train_unscene3d.sh "${experiments_base_name}_4" "trainer.max_epochs=70 optimizer=adamw_lower data.train_dataset.self_train_data_dir=${self_train_base_path}/${experiments_base_name}_3 general.checkpoint=${self_train_base_path}/${experiments_base_name}_3 data.train_dataset.load_self_train_data=true"
python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} general.experiment_name="${experiments_base_name}_4" general.checkpoint=true
mv saved/${experiments_base_name}_4 ${self_train_base_path}/${experiments_base_name}_4