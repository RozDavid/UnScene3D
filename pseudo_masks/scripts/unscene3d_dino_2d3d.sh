python unscene3d_pseudo_main.py \
        misc.comment=cutler3d_CSC_DINO \
        freemask.affinity_tau=0.6 \
        freemask.modality=both \
        data.segments_min_vert_nums=[50] \
        freemask.min_segment_size=4 \
        test.visualize=true \
        +save_dir=outputs/