#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import pickle
import numpy as np
import sys
import logging
import argparse
import cv2

from backbone_fc import EmbedNetwork
from datasets.VehicleID import VehicleID
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--store_pth',
            dest = 'store_pth',
            type = str,
            default='./res/soft_trip_res50/emb_25000.pkl',
            help = 'path that the embeddings are stored: e.g.: ./res/emb.pkl',
            )
    parse.add_argument(
            '--model',
            dest = 'model',
            type = str,
            default='./res/soft_trip_res50/25000model_trip_soft.pkl',
            help = 'path that the embeddings are stored: e.g.: ./res/emb.pkl',
            )
    parse.add_argument(
            '--data_pth',
            dest = 'data_pth',
            type = str,
            default='/home/CORP/ryann.bai/dataset/VehicleID/image/',
            help = 'path that the raw images are stored',
            )
    parse.add_argument(
            '--data_list',
            dest = 'data_list',
            type = str,
            default='/home/CORP/ryann.bai/dataset/VehicleID/train_test_split_v1/test_list_800.txt',
            help = ' the raw images list',
            )
    parse.add_argument('--model_name', dest='model_name',
                    help='model_name',
                    default='vggm', type=str)
    parse.add_argument( '--num_class', default=13164, type=int,
    help='The numberK used in the PK-batches')
    parse.add_argument( '--img_size', default=224, type=int,
    help='The numberK used in the PK-batches')
    return parse.parse_args()



def embed(args):
    ## logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## restore model
    logger.info('restoring model')
    if args.model_name == 'vggm':
        logger.info('model select vggm')
        model = vggm(num_class = args.num_class).cuda()
    else:
        logger.info('model select resnet50')
        model = EmbedNetwork(num_class = args.num_class).cuda()
        
    #model = EmbedNetwork().cuda()
    model.load_state_dict(torch.load(args.model))
    model = nn.DataParallel(model)
    model.eval()

    ## load gallery dataset
    batchsize = 24
    ds = VehicleID(args.data_pth, args.data_list,img_size=args.img_size, is_train = False)
    dl = DataLoader(ds, batch_size = batchsize, drop_last = False, num_workers = 4)

    ## embedding samples
    logger.info('start embedding')
    all_iter_nums = len(ds) // batchsize + 1
    embeddings = []
    label_ids = []
    label_cams = []
    img_names = []
    for it, (img, lb_id, lb_cam, img_name) in enumerate(dl):
        print('\r=======>  processing iter {} / {}'.format(it, all_iter_nums),
                end = '', flush = True)
        label_ids.append(lb_id)
        label_cams.append(lb_cam)
        img_names.append(img_name)
        embds = []
        for im in img:
            im = im.cuda()
            embd, _ = model(im)
            embd = embd.detach().cpu().numpy()
            embds.append(embd)
        embed = sum(embds) / len(embds)
        embeddings.append(embed)
    print('  ...   completed')

    embeddings = np.vstack(embeddings)
    label_ids = np.hstack(label_ids)
    label_cams = np.hstack(label_cams)
    img_names = np.hstack(img_names)

    ## dump results
    logger.info('dump embeddings')
    embd_res = {'embeddings': embeddings, 'label_ids': label_ids, 'label_cams': label_cams, 'img_names':img_names}
    with open(args.store_pth, 'wb') as fw:
        pickle.dump(embd_res, fw)

    logger.info('embedding finished')


if __name__ == '__main__':
    args = parse_args()
    embed(args)
