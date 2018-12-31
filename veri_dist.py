import pickle
import numpy as np
import sys
import logging
import argparse

parser = argparse.ArgumentParser(description='Train a verification model')
parser.add_argument('--query_embeddings', dest='query_embeddings', 
                    default='./res/trip_vggm_VeRI/emb_25000_query.pkl', type=str,
                    help='Path to the h5 file containing the query embeddings.')
parser.add_argument('--query_list_file', dest='query_list_file',
                    help='test list file',
                    default='/home/CORP/ryann.bai/dataset/VeRi-776/name_query.txt', type=str)
parser.add_argument('--ref_embeddings', dest='ref_embeddings', 
                    default='./res/trip_vggm_VeRI/emb_25000_ref.pkl', type=str,
                    help='Path to the h5 file containing the query embeddings.')
parser.add_argument('--ref_list_file', dest='ref_list_file',
                    help='test list file',
                    default='/home/CORP/ryann.bai/dataset/VeRi-776/name_test.txt', type=str)
parser.add_argument('--dist_file', dest='dist_file',
                    help='dist_file',
                    default='./res/trip_vggm_VeRI/dist_25000.txt', type=str)
args = parser.parse_args()
query = []
gallery = []
q_n = 0
g_n = 0

for line in open(args.query_list_file).readlines():
    t = line.strip().split(' ')[0]
    query.append(t)
    q_n = q_n + 1
    
    #print q_n
for line in open(args.ref_list_file).readlines():
    t = line.strip().split(' ')[0]
    gallery.append(t)
    g_n = g_n + 1
print('read all')

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
    r_embs_dict[name] = emb
print('load emb finish')    
   
dist = np.zeros([g_n, q_n], dtype=np.float32)

fout = open(args.dist_file, 'w')
for g in range(0,g_n):
    for q in range(0, q_n):
        g_name = gallery[g]
        g_feat = r_embs_dict[g_name]
        q_name = query[q]
        q_feat = q_embs_dict[q_name]
        dist[g][q] = np.linalg.norm(g_feat-q_feat)
        fout.writelines(str(dist[g][q]) +' ')
    fout.writelines('\n')
fout.close()
print('over')
