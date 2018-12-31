#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging
import time
import itertools

from backbone_fc import EmbedNetwork
from VGGM import vggm,VGGM
from loss import TripletLoss
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from datasets.VehicleID import VehicleID
from optimizer import AdamOptimWrapper
from logger import logger
from torch.autograd import Variable

import argparse

parser = argparse.ArgumentParser(description='Train a verification model')
parser.add_argument('--dataset', dest='dataset', 
                    default='VehicleID', type=str,
                    help='dataset')
parser.add_argument('--img_dir', dest='img_dir',
                    help='img_dir',
                    default='/home/CORP/ryann.bai/dataset/VehicleID/image/', type=str)
parser.add_argument('--img_list', dest='img_list',
                    help='img_list',
                    default='/home/CORP/ryann.bai/dataset/VehicleID/train_test_split_v1/train_list_start0_jpg.txt', type=str)
parser.add_argument('--model', dest='model',
                    help='model',
                    default='vggm', type=str)
parser.add_argument('--save_path', dest='save_path',
                    help='save_path',
                    default='./res/soft_trip_vggm/', type=str)
parser.add_argument('--model_name', dest='model_name',
                    help='model_name',
                    default='model_trip_vggm.pkl', type=str)
parser.add_argument('--learning_rate', default=3e-4, type=float,
    help='The initial value of the learning-rate, before it kicks in.')

parser.add_argument('--train_iterations', default=25000, type=int,
    help='Number of training iterations.')

parser.add_argument('--decay_start_iteration', default=15000, type=int,
    help='At which iteration the learning-rate decay should kick-in.'
         'Set to -1 to disable decay completely.')

parser.add_argument('--checkpoint_frequency', default=5000, type=int,
    help='After how many iterations a checkpoint is stored. Set this to 0 to '
         'disable intermediate storing. This will result in only one final '
         'checkpoint.')
parser.add_argument('--batch_p', default=18, type=int,
    help='The number P used in the PK-batches')

parser.add_argument( '--batch_k', default=4, type=int,
    help='The numberK used in the PK-batches')
parser.add_argument( '--num_class', default=13164, type=int,
    help='The numberK used in the PK-batches')
parser.add_argument( '--resume', default='', type=str,
    help='The numberK used in the PK-batches')
args = parser.parse_args()
save_path = args.save_path
model_name = args.model_name
dataset = args.dataset
img_dir = args.img_dir
img_list = args.img_list
model = args.model
def train():
    ## setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists(save_path): os.makedirs(save_path)
        

    ## model and loss
    logger.info('setting up backbone model and loss')
    if model == 'vggm':
        logger.info('model select vggm')
        net = vggm(num_class = args.num_class).cuda()
    else:
        logger.info('model select resnet50')
        net = EmbedNetwork(num_class = args.num_class).cuda()
    if(args.resume != ''):
        net.load_state_dict(torch.load(args.resume))
        logger.info('fine-turn from {}'.format(args.resume))
    net = nn.DataParallel(net)
    triplet_loss = TripletLoss(margin = None).cuda() # no margin means soft-margin
    
    softmax_criterion = torch.nn.CrossEntropyLoss()

    ## optimizer
    logger.info('creating optimizer')
    optim = AdamOptimWrapper(net.parameters(), lr = args.learning_rate, wd = 0, t0 = args.decay_start_iteration, t1 = args.train_iterations)

    ## dataloader
    selector = BatchHardTripletSelector()
    
    ds = VehicleID(img_dir, img_list, img_size = 256, is_train = True)
    logger.info('dataset OK')  
    sampler = BatchSampler(ds, args.batch_p, args.batch_k)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    diter = iter(dl)

    ## train
    logger.info('start training ...')
    loss_avg = []
    loss_soft_avg = []
    count = 0
    t_start = time.time()
    while True:
        try:
            imgs, lbs, _, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, lbs, _, _ = next(diter)

        net.train()
        imgs = imgs.cuda()
        lbs = lbs.cuda()
        if model == 'vggm':
            embds, fc = net(imgs)
        else:
            embds, fc = net(imgs)
        anchor, positives, negatives = selector(embds, lbs)

        loss = triplet_loss(anchor, positives, negatives)
        loss_softmax = softmax_criterion(fc,lbs)
        
        loss_all = 0.5 * loss_softmax + loss
        optim.zero_grad()
        loss_all.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        loss_soft_avg.append(loss_softmax.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            loss_soft_avg = sum(loss_soft_avg) / len(loss_soft_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('iter: {}, trip_loss: {:4f}, soft_loss: {:4f}, lr: {:4f}, time: {:3f}'.format(count, loss_avg, loss_soft_avg, optim.lr, time_interval))
            loss_avg = []
            loss_soft_avg = []
            t_start = t_end
            
        if count % args.checkpoint_frequency ==0 and count != 0:
            logger.info('saving trained model')
            name = save_path + str(count) + model_name
            ver = 2
            while(os.path.exists(name)):
                logger.info('model has exists')
                name = name + '_v'+str(ver)
                ver = ver + 1
            torch.save(net.module.state_dict(), name)

        count += 1
        if count == args.train_iterations: break

    ## dump model
    logger.info('saving trained model')
    name = save_path + str(count) + '_'+ model_name
    ver = 2
    while(os.path.exists(name)):
        logger.info('model has exists')
        name = name + '_v'+str(ver)
        ver = ver + 1
    torch.save(net.module.state_dict(), name)
    logger.info('everything finished')


if __name__ == '__main__':
    train()
