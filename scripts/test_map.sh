#!/bin/bash
    
python ../mAP_vehicleID.py \
    --embeddings ../res/soft_trip_res50_VehicleID/emb_15000_1600_v2.pkl \
    --repeat 2 \
    --save ../res/soft_trip_res50_VehicleID/map_resnet50.txt \
    --list_file /home/CORP/ryann.bai/dataset/VehicleID/train_test_split_v1/test_list_1600.txt

python ../mAP_vehicleID.py \
    --embeddings ../res/soft_trip_res50_VehicleID/emb_10000_1600_v2.pkl \
    --repeat 2 \
    --save ../res/soft_trip_res50_VehicleID/map_resnet50.txt \
    --list_file /home/CORP/ryann.bai/dataset/VehicleID/train_test_split_v1/test_list_1600.txt