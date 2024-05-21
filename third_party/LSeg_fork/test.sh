#export CUDA_VISIBLE_DEVICES=0; python test_lseg.py --backbone clip_vitl16_384 --eval --dataset ade20k --data-path ../datasets/ \
#--weights checkpoints/lseg_ade20k_l16.ckpt --widehead --no-scaleinv


export CUDA_VISIBLE_DEVICES=0;
python test_lseg.py --backbone clip_vitl16_384 \
--eval \
--dataset ade20k \
--data-path /mnt/cluster/himring/drozenberszki/Datasets/ScanNet2D/subset \
--weights checkpoints/demo_e200.ckpt \
--widehead \
--no-scaleinv



