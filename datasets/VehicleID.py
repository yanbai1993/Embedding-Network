#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from logger import logger
from random_erasing import RandomErasing


class VehicleID(Dataset):
    '''
    a wrapper of VehicleID dataset
    '''
    def __init__(self, data_path, train_file, img_size, is_train = True, *args, **kwargs):
        super(VehicleID, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.check = os.listdir(data_path)
        self.check = [el for el in self.check if os.path.splitext(el)[1] == '.jpg']
        self.dict = {}
        f = open(train_file, 'r')
        self.train_file = f.readlines()
        self.imgs = []
        max_class = 0
        self.classes = set()
        now = 0
        for line in self.train_file:
            line=line.strip()
            if(len(line.split(' ')) ==2):
                car_name,car_class = line.split(' ')[0],int(line.split(' ')[1])
            elif(len(line.split(' ')) ==1 and not is_train):
                car_name,car_class = line.split(' ')[0], -1
            else:
                logger.info('dataset wrong')
            #if car_name not in self.check:
            #    logger.warning("%s do not exists"%(car_name))
            #    continue
            if(len(car_name.split('.')) == 1):
                car_name = car_name + '.jpg'
            if car_name not in self.dict:
                self.dict[car_name] = car_class
                max_class = max(max_class, int(car_class))
                self.classes.add(car_class)
                self.imgs.append(car_name)
                
        logger.info('dataset OK')        
        if self.is_train:
            assert(max_class == len(self.classes)-1)    
        self.lb_ids = [self.dict[el] for el in self.imgs]
        self.lb_cams = [-1 for el in self.imgs]
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]
        if is_train:
            self.trans = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
                RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
            ])
        else:
            self.trans_tuple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
                ])
            self.Lambda = transforms.Lambda(
                lambda crops: [self.trans_tuple(crop) for crop in crops])
            self.trans = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.TenCrop((img_size, img_size)),
                self.Lambda,
            ])

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = self.trans(img)
        return img, self.lb_ids[idx], self.lb_cams[idx], self.imgs[idx]



if __name__ == "__main__":
    img_dir = "/home/CORP/ryann.bai/dataset/VehicleID/image/"
    img_list = "/home/CORP/ryann.bai/dataset/VehicleID/train_test_split_v1/train_list_start0_jpg.txt"
    ds = VehicleID(img_dir, img_list, is_train = True)
    im, _, _ = ds[1]
    print(im.shape)
    print(im.max())
    print(im.min())
    ran_er = RandomErasing()
    im = ran_er(im)
    cv2.imshow('erased', im)
    cv2.waitKey(0)
