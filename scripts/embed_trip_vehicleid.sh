#!/bin/bash

python ../embed_vehicleID_fc.py \
    --store_pth ../res/soft_trip_res50_VehicleID/emb_10000_800_v2.pkl \
    --model ../res/soft_trip_res50_VehicleID/10000model_trip_soft_res50_v2.pkl \
    --data_pth VehicleID/image/ \
    --data_list VehicleID/train_test_split_v1/test_list_800.txt \
    --model_name res50 \
    --num_class 13164 \
    --img_size 256

python ../embed_vehicleID_fc.py \
    --store_pth ../res/soft_trip_res50_VehicleID/emb_10000_2400_v2.pkl \
    --model ../res/soft_trip_res50_VehicleID/10000model_trip_soft_res50_v2.pkl \
    --data_pth VehicleID/image/ \
    --data_list VehicleID/train_test_split_v1/test_list_2400.txt \
    --model_name res50 \
    --num_class 13164 \
    --img_size 256
    
python ../embed_vehicleID_fc.py \
    --store_pth ../res/soft_trip_res50_VehicleID/emb_10000_1600_v2.pkl \
    --model ../res/soft_trip_res50_VehicleID/10000model_trip_soft_res50_v2.pkl \
    --data_pth VehicleID/image/ \
    --data_list VehicleID/train_test_split_v1/test_list_1600.txt \
    --model_name res50 \
    --num_class 13164 \
    --img_size 256

     
    
    #--data_pth /home/CORP/ryann.bai/dataset/VehicleID/image/ \
    #--data_list /home/CORP/ryann.bai/dataset/VehicleID/train_test_split_v1/test_list_800.txt \
    #--data_pth /home/CORP/ryann.bai/dataset/VeRi-776/image_test/ \
    #--data_list /home/CORP/ryann.bai/dataset/VeRi-776/name_test.txt \