import os
import numpy as np
import pickle
from tqdm import tqdm
from argparse import ArgumentParser

from eval import eval


parser = ArgumentParser(description='face model test')
parser.add_argument('--mode', default='val')
parser.add_argument('--root', default='data')
parser.add_argument('--gt', type=str, default='val_label.json')
parser.add_argument('--submission', type=str, default='result.txt')

parser.add_argument('--face_rerank', default=False, action='store_true')
parser.add_argument('--body_rerank', default=False, action='store_true')

parser.add_argument('--thres', default=[0.5, 0.4, 0.3, 0.25], type=float, nargs='+')
args = parser.parse_args()

args.submission = '{}_{}'.format(args.mode, args.submission)
args.gt = '{}/{}'.format(args.root, args.gt)
args.runs = len(args.thres) - 1


def load_pkl(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
        return data


def unique_list(lst):
    unique_lst = []
    val_set = set()
    for x in lst:
        if x not in val_set:
            unique_lst.append(x)
            val_set.add(x)
    return unique_lst
# Using only body features to determine the ranking list of the given probe
def decide():

    body_dict = load_pkl('{}/body_{}.pkl'.format(args.root, args.mode))
    
    reid_dicts = [load_pkl('{}/triplet_resnext101_32x8d_{}.pkl'.format(args.root, args.mode))]

    print('load data done')

    result = {}
    for movie in tqdm(sorted(reid_dicts[0].keys())):
        cast = body_dict[movie]['cast']
        candi = body_dict[movie]['candidates']
        num_cast = len(cast)
        num_candi = len(candi)
        print("Number of candi ",num_candi)
        print("Number of cast ",num_cast)
        cast_ids = np.array([item['id'] for item in cast])
        candi_ids = np.array([item['id'] for item in candi])

        candi_files = np.array([os.path.basename(item['file']).split('_')[0] for item in candi])
        file_cnt = {}
        for file in candi_files:
            if file not in file_cnt:
                file_cnt[file] = 1
            else:
                file_cnt[file] += 1
        candi_file_cnt = np.array([file_cnt[file] for file in candi_files])

        result.update({cast_id: [] for cast_id in cast_ids})

        embs = [reid_dict[movie] for reid_dict in reid_dicts]

        embs = [emb / np.linalg.norm(emb, axis=-1, keepdims=True) for emb in embs]
        cast_embs = embs[0][:num_cast]
        candi_embs = embs[0][num_cast:]

        # print(len(candi_embs))
        # print(len(candi_embs[0]))
        cast_candi_sim = np.dot(cast_embs, candi_embs.T)
        candi_candi_sim = np.dot(candi_embs, candi_embs.T)

        
        sort_index = np.argsort(1 - cast_candi_sim, axis=-1)
        match = cast_candi_sim > args.thres[0]
        match_num = match.sum(1)
        total_match = match
        idx = np.where(match_num < 5)[0]
        if len(idx) > 0:
            match[idx[:, np.newaxis], sort_index[idx, :5]] = True
            match_num[idx] = 5

        for i in range(num_cast):
            result[cast_ids[i]].extend(sort_index[i][:match_num[i]])

        for run in range(args.runs):
            sims = []
            for i in range(num_cast):
                query = np.where(total_match[i])[0]
                sim = candi_candi_sim[query].mean(0)
                sims.append(sim)

            cast_candi_sim = np.stack(sims)
            sort_index = np.argsort(1 - cast_candi_sim, axis=-1)
            match = cast_candi_sim > args.thres[run + 1]
            match_num = match.sum(1)
            total_match = np.logical_or(match, total_match)

            for i in range(num_cast):
                prev = result[cast_ids[i]]
                cur = unique_list(prev + list(sort_index[i][:match_num[i]]))
                result[cast_ids[i]] = cur

        match = total_match
        for i, cast_id in enumerate(cast_ids):
            vals = result[cast_id]
            vals = unique_list(vals)
            result[cast_id] = [candi_ids[j] for j in vals]

    with open(args.submission, 'w') as f:
        for key, vals in result.items():
            f.writelines('{} {}\n'.format(key, ','.join(vals)))



if __name__ == '__main__':
    decide()
    if args.mode == 'val':
        eval(args.submission, args.gt)
