#!/bin/bash
python ../train_soft_trip_v2.py \
       --dataset VehicleID \
       --img_dir VehicleID/image/  \
       --img_list VehicleID/train_test_split_v1/train_list_start0_jpg.txt \
       --model res50 \
       --save_path ../res/soft_trip_res50_VehicleID/ \
       --model_name model_trip_soft_res50_v2.pkl \
       --batch_p 18 \
       --learning_rate 3e-4 \
       --train_iterations 50000 \
       --decay_start_iteration 25000\
       --num_class 13164 \
       #--resume ../res/soft_trip_res50_VehicleID/25000model_trip_soft.pkl
       
       