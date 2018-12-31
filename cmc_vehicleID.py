import pickle
import numpy as np
import sys
import logging
import argparse


parser = argparse.ArgumentParser(description='CMC')
parser.add_argument('--embeddings', dest='embeddings', 
                    default='./res/soft_trip_res50_VehicleID/emb_25000.pkl', type=str,
                    help='Path to the h5 file containing the query embeddings.')
parser.add_argument('--list_file', dest='list_file',
                    help='test list file',
                    default='test_list_800.txt', type=str)
parser.add_argument('--repeat', dest='repeat',
                    help='repeat times',
                    default=10, type=int)
parser.add_argument('--save', dest='save',
                    help='save to file',
                    default='./res/trip_res50/cmc_log.txt', type=str)

args = parser.parse_args()


def load_query_reference(imagelist_file,query_file):
    gallery = {}
    probe = {}
    for line in open(imagelist_file).readlines():
        line = line.strip()
        t = line.split('/')
        if int(t[0]) not in gallery:
            gallery[int(t[0])] = []
        gallery[int(t[0])].append(line)
    for line in open(query_file).readlines():
        line = line.strip()
        t = line.split('/')
        if int(t[0]) not in probe:
            probe[int(t[0])] = []
        probe[int(t[0])].append(line)
    return gallery, probe

def load_cls_samples(fname):
    cls_samples = {}
    for line in open(fname).readlines():
        t = line.strip().split()
        if int(t[1]) not in cls_samples:
            cls_samples[int(t[1])] = []
        cls_samples[int(t[1])].append(t[0])
    return cls_samples

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



RANK_LIST = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
with open(args.embeddings, 'rb') as fr:
    embeddings_dict = pickle.load(fr)
    embs, lb_ids, lb_cams, img_names = embeddings_dict['embeddings'], embeddings_dict['label_ids'], embeddings_dict['label_cams'], embeddings_dict['img_names']

embs_dict = {}
for index in range(len(lb_ids)):
    ids = lb_ids[index]
    emb = embs[index]
    name = img_names[index].split('/')[-1].split('.')[0]
    embs_dict[name] = emb
FEAT_SIZE = embs.shape[1]

mean_avg_cmc = []
average_rank_rate = np.zeros([len(RANK_LIST), ])
for r_id in range(args.repeat):
    cls_samples = load_cls_samples(args.list_file)
    gallery, probe = gen_gallery_probe(cls_samples)
    
    if r_id==0:
        print('Gallery size: %d' % (len(gallery.keys())))
    gids = []

    gids = list(gallery.keys())

    g_feat = np.zeros([len(gids), FEAT_SIZE], dtype=np.float32)
    for i in range(0,len(gids)):
        name = gallery[gids[i]][0]
        fea = embs_dict[name]
        g_feat[i] = fea

    if r_id==0:
        print('Gallery feature extraction finished')

    rank_rate = np.zeros([len(RANK_LIST), ])
    cnt = 0
    for pid in probe:
        for psample in probe[pid]:
            #  gids = gallery.keys()
            #print pid, psample
            g_dist = np.zeros([len(gids),])
            p_feat = np.zeros([FEAT_SIZE,], dtype=np.float32)
            fea = embs_dict[psample]
            p_feat = fea

            for i in range(0,len(gids)):
                g_dist[i] = np.linalg.norm(g_feat[i]-p_feat)
                
            g_sorted = [gids[i] for i in g_dist.argsort()]
            for k, r in enumerate(RANK_LIST):
                if pid in g_sorted[:r]:
                    rank_rate[k] += 1

            cnt += 1
            #  print '%s finished(%d)' % (psample, cnt), rank_rate/cnt

    rank_rate /= cnt
    print('============================= ITERATION %d =============================' % (r_id+1))
    print(RANK_LIST)
    print(rank_rate)
    mean_avg_cmc.append(rank_rate)
    #  print '========================================================================'
    average_rank_rate += rank_rate
average_rank_rate /= args.repeat
print('Average rank rate: ')
print(average_rank_rate)
if args.save != '':
    with open(args.save, 'a') as fd:
        fd.write("list_file '{}' \n".format(args.list_file))
        fd.write("embeddings '{}' \n".format(args.embeddings))
        for r_id in range(0,args.repeat):
            fd.write('ITERATION %d: %s\n' % (r_id+1, str(mean_avg_cmc[r_id])))
        fd.writelines(str(average_rank_rate)+'\n')
