import torch

import pickle
import numpy as np
import sys
import logging
import argparse

parser = argparse.ArgumentParser(description='Train a verification model')
parser.add_argument('--query_embeddings', dest='query_embeddings', 
                    default='./res/trip_vggm_VeRI/emb_25000_query.pkl', type=str,
                    help='Path to the h5 file containing the query embeddings.')
parser.add_argument('--ref_embeddings', dest='ref_embeddings', 
                    default='./res/trip_vggm_VeRI/emb_25000_ref.pkl', type=str,
                    help='Path to the h5 file containing the query embeddings.')
parser.add_argument('--list_file', dest='list_file',
                    help='test list file',
                    default='', type=str)
parser.add_argument('--query_file', dest='query_file',
                    help='query list file',
                    default='', type=str)
parser.add_argument('--repeat', dest='repeat',
                    help='repeat times',
                    default=1, type=int)
parser.add_argument('--maxg', dest='maxg',
                    help='max number of a class id in gallery',
                    default=1000, type=int)
parser.add_argument('--save', dest='save',
                    help='save to file',
                    default='./res/trip_res50/map_resnet50.txt', type=str)

args = parser.parse_args()

def load_query_reference(imagelist_file,query_file):
    gallery = {}
    probe = {}
    for line in open(imagelist_file).readlines():
        line = line.strip()
        t = line.split('/')
        if int(t[0]) not in gallery:
            gallery[int(t[0])] = []
        gallery[int(t[0])].append(t[1])
    for line in open(query_file).readlines():
        line = line.strip()
        t = line.split('/')
        if int(t[0]) not in probe:
            probe[int(t[0])] = []
        probe[int(t[0])].append(t[1])
    return gallery, probe


def gen_gallery_probe(samples, k=1):
    """
    k: k samples for each id in gallery, 1 for reid
    TODO: generate gallery and probe sets.
    """
    cls_ids = samples.keys()
    gallery = {}
    probe = {}
    for cls_id in cls_ids:
        cls_samples = samples[cls_id]
        if len(cls_samples)<=1:
            continue
        gallery[cls_id] = []
        probe[cls_id] = []
        n = len(cls_samples)
        #  gid = np.random.randint(0, n)
        gids = np.random.permutation(np.arange(n))[:min(n-1, k)]
        for i in range(len(cls_samples)):
            if i in gids:
                gallery[cls_id].append(cls_samples[i])
            else:
                probe[cls_id].append(cls_samples[i])
    return gallery, probe

def load_cls_samples(fname):
    cls_samples = {}
    for line in open(fname).readlines():
        t = line.strip().split()
        if int(t[1]) not in cls_samples:
            cls_samples[int(t[1])] = []
        cls_samples[int(t[1])].append(t[0])
    return cls_samples

mean_avg_prec = np.zeros([args.repeat, ])
mean_avg_prec_p = np.zeros([args.repeat, ])
mean_avg_prec_e = np.zeros([args.repeat, ]) 

    
with open(args.query_embeddings, 'rb') as fr:
    q_embeddings_dict = pickle.load(fr)
    embs, lb_ids, lb_cams, img_names = q_embeddings_dict['embeddings'], q_embeddings_dict['label_ids'], q_embeddings_dict['label_cams'], q_embeddings_dict['img_names']

q_embs_dict = {}
for index in range(len(lb_ids)):
    emb = embs[index]
    name = img_names[index].split('/')[-1]
    q_embs_dict[name] = emb
   
with open(args.ref_embeddings, 'rb') as fr:
    r_embeddings_dict = pickle.load(fr)
    embs, lb_ids, lb_cams, img_names = r_embeddings_dict['embeddings'], r_embeddings_dict['label_ids'], r_embeddings_dict['label_cams'], r_embeddings_dict['img_names']

r_embs_dict = {}
for index in range(len(lb_ids)):
    emb = embs[index]
    name = img_names[index].split('/')[-1]
    #print(len(emb))
    r_embs_dict[name] = emb    
    
FEAT_SIZE = embs.shape[1]

for r_id in range(args.repeat):
    gallery, probe = load_query_reference(args.list_file, args.query_file)

    if r_id==0:
        print('Gallery size: %d' % (len(gallery.keys())))     
    g_n = 0
    p_n = 0
    for gid in gallery:
        g_n += len(gallery[gid])
    for pid in probe:
        p_n += len(probe[pid])
    print(g_n,p_n)
    g_feat = np.zeros([g_n, FEAT_SIZE], dtype=np.float32)
    g_ids = np.zeros([g_n, ], dtype=np.float32)
    g_imgs = []
    k = 0
    for gid in gallery.keys():
        for s in gallery[gid]:
           
            if(s not in r_embs_dict):
                if(len(s.split('.')) == 1):
                    s = s + '.jpg'
                else:
                    s = s.split('.')[0]
            
            assert(s in r_embs_dict)
            fea = r_embs_dict[s]
            g_feat[k] = fea
            g_ids[k] = gid
            g_imgs.append(s)
            k += 1

    if r_id==0:
        print('Gallery feature extraction finished')

    for pid in probe:# for every probe only has one sample
        #probe
        psample = probe[pid][0]
        p_dist = np.zeros([g_n,], dtype=np.float32)
        if(psample not in q_embs_dict):
            if(len(psample.split('.')) == 1):
                psample = psample + '.jpg'
            else:
                psample = psample.split('.')[0]
        assert(psample in q_embs_dict)
        fea = q_embs_dict[psample]
        p_feat = fea

        ##compute distance
        dist = np.zeros([g_n,], dtype=np.float32)
        for i in range(g_n):
            p_dist[i] = np.linalg.norm(g_feat[i]-p_feat)

        ##probe
        p_sorted = np.array([g_ids[i] for i in p_dist.argsort()])
        p_sortimg = np.array([g_imgs[i] for i in p_dist.argsort()])
        n = np.sum(p_sorted==pid)
        hit_inds_p = np.where(p_sorted==pid)[0]
        map_p = 0
        for i, ind in enumerate(hit_inds_p):
            map_p += (i+1)*1.0/(ind+1)
        map_p /= n
        mean_avg_prec_p[r_id] += map_p
    mean_avg_prec_p[r_id] /= p_n
    print("query_embeddings '{}' \n".format(args.query_embeddings))
    print("ref_embeddings '{}' \n".format(args.ref_embeddings))
    print('============================= ITERATION %d =============================' % (r_id+1))
    print(mean_avg_prec_p[r_id])
    

print('Average MAP:', np.mean(mean_avg_prec_p))
if args.save != '':
    with open(args.save, 'a') as fd:
        fd.write("list_file '{}' \n".format(args.list_file))
        fd.write("query_embeddings '{}' \n".format(args.query_embeddings))
        fd.write("ref_embeddings '{}' \n".format(args.ref_embeddings))
        for r_id in range(args.repeat):
            fd.write('ITERATION %d: %.6f\n' % (r_id+1, mean_avg_prec_p[r_id]))
        fd.write('mean ave mAP %.6f\n' % np.mean(mean_avg_prec_p))
