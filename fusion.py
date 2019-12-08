import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn import metrics
from argparse import ArgumentParser
from scipy.sparse import csr_matrix
from eval import eval
from ecn import ECN
from kreciprocal import reciprocal

parser = ArgumentParser(description='face model test')
parser.add_argument('--mode', default='val')
parser.add_argument('--root', default='data')
parser.add_argument('--gt', type=str, default='val_label.json')
parser.add_argument('--submission', type=str, default='result.txt')

parser.add_argument('--face_rerank', default=False, action='store_true')
parser.add_argument('--body_rerank', default=False, action='store_true')
parser.add_argument('--body_rerank_ecn', default=False, action='store_true')
parser.add_argument('--face_rerank_ecn', default=False, action='store_true')
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

def decide():

    # Combined results of face detection: movie-> {cast: [{face,landmark,score,label,id}..], candidates:[{face,landmark,score,label,id}..]}
    body_dict = load_pkl('{}/retina_{}.pkl'.format(args.root, args.mode))
    # Movie's cast and candidate facial features
    face_dict = load_pkl('{}/face_{}.pkl'.format(args.root, args.mode))
    # Movie's candidate's body features
    reid_dicts = [load_pkl('{}/triplet_resnext101_32x8d_{}.pkl'.format(args.root, args.mode))]
    result = {}
    for movie in tqdm(sorted(body_dict.keys())):
        cast = body_dict[movie]['cast']
        candi = body_dict[movie]['candidates']
        num_cast = len(cast)
        num_candi = len(candi)

        cast_ids = np.array([item['id'] for item in cast])
        candi_ids = np.array([item['id'] for item in candi])

        # All candidate images of a given movie
        candi_files = np.array([os.path.basename(item['file']).split('_')[0] for item in candi])
        file_cnt = {}
        for file in candi_files:
            if file not in file_cnt:
                file_cnt[file] = 1
            else:
                file_cnt[file] += 1
        candi_file_cnt = np.array([file_cnt[file] for file in candi_files])
        # True if the face detection algorithm, RetinaFace did not recognize the face in a candidate image
        no_face = np.array([item['faces'] is None for item in candi])

        result.update({cast_id: [] for cast_id in cast_ids})
        #Features of the current movie
        feats = face_dict[movie]
        feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
        # k-reciprocal encoding 
        if args.face_rerank:
            sim = np.dot(feats, feats.T)
            dists = np.sqrt(2 - 2 * sim + 1e-4)
            dists = reciprocal(dists, num_cast)
            cast_candi_sim = 1 - dists
        # expanded cross neighborhood re-ranking
        elif args.face_rerank_ecn:
            ranked_list_of_ecn = ECN(feats, num_cast, num_candi)
            cast_candi_sim = cast_candi_sim = 1-ranked_list_of_ecn     
        else:
            cast_candi_sim = np.dot(feats[:num_cast], feats[num_cast:].T)
        candi_candi_sim = np.dot(feats[num_cast:], feats[num_cast:].T)
        
        cast_candi_sim[:, no_face] = -2
        candi_candi_sim[:, no_face] = -2
        candi_candi_sim[no_face, :] = -2

        sort_index = np.argsort(1 - cast_candi_sim, axis=-1)
        # If the candidate's similarity with the cast is higher than the given threshold - 0.5 ->TRUE
        match = cast_candi_sim > args.thres[0]
        match_num = match.sum(1)
        total_match = match
        idx = np.where(match_num < 5)[0]
        if len(idx) > 0:
            match[idx[:, np.newaxis], sort_index[idx, :5]] = True
            match_num[idx] = 5

        # Adding the initial candidate list
        for i in range(num_cast):
            result[cast_ids[i]].extend(sort_index[i][:match_num[i]])

        # Expanding the match set by lowering the threshold
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

        # Candidates for whom the cast is decided
        decided = np.where(match.sum(0))[0]
        decided = list(decided)
        decided_set = set(decided)
        unknown = [i for i in range(num_candi) if i not in decided_set]

        embs = [reid_dict[movie] for reid_dict in reid_dicts]
        embs = [emb / np.linalg.norm(emb, axis=-1, keepdims=True) for emb in embs]
        embs = np.concatenate(embs, axis=-1)
        embs /= np.linalg.norm(embs, axis=-1, keepdims=True)
        for reid_dict in reid_dicts:
            del reid_dict[movie]

        qembs = embs
        gembs = embs[np.array(unknown)]

        # If k-reciprocal encoding is enabled
        if args.body_rerank:
            emb = np.concatenate([qembs, gembs], axis=0)
            body_sim = np.dot(emb, emb.T)
            dists = np.sqrt(2 - 2 * body_sim + 1e-4)
            dists = reciprocal(dists, len(qembs))
            body_sim = 1 - dists
        elif args.body_rerank_ecn:
            emb = np.concatenate([qembs, gembs], axis=0)
            body_sim = 1-ECN(emb, len(qembs), len(gembs))    
        else:
            body_sim = np.dot(qembs, gembs.T)

        origin_sim = np.dot(embs, embs.T)
        # Taking the first 9 neighbors for averaging the feature values
        knn = np.argpartition(1 - origin_sim, 9, axis=-1)[:, :9]

        knn_sim = origin_sim[np.arange(len(embs))[:, np.newaxis], knn]
        exp_knn_sim = np.exp(knn_sim)
        # Calcuting the weight using softmax function
        weight = exp_knn_sim / exp_knn_sim.sum(axis=-1, keepdims=True)

        knn_embs = embs[knn.reshape(-1)].reshape(len(embs), 9, -1)
        knn_embs = (knn_embs * weight[:, :, np.newaxis]).sum(axis=1)
        knn_embs /= np.linalg.norm(knn_embs, axis=-1, keepdims=True)

        knn_qembs = knn_embs
        knn_gembs = knn_embs[np.array(unknown)]
        # Finding the similarity matrix between decided and undecided set
        knn_body_sim = np.dot(knn_qembs, knn_gembs.T)

        body_sim = (body_sim + knn_body_sim) / 2

        decide_sim = np.dot(embs, embs[np.array(decided)].T)

        # Add similar candidate of a cast as well as candidates similar to each of those candidates to the result
        for i in range(num_cast):
            # Expanding the result set
            ind = np.unique(result[cast_ids[i]])
            sim = body_sim[ind]
            if len(ind) > 3:
                sim = sim[np.argpartition(1 - sim, 3, axis=0), np.arange(len(unknown))][:3]
            exp_sim = np.exp(sim)
            weight = exp_sim / exp_sim.sum(0)
            sim = (sim * weight).sum(0)

            end = []
            idx = np.where(candi_file_cnt[ind] > 1)[0]
            for index in ind[idx]:
                jj = np.where(candi_files == candi_files[index])[0]
                end.extend(jj)
            end = set(end)

            for j in np.argsort(1 - sim):
                jj = unknown[j]
                if jj not in end:
                    result[cast_ids[i]].append(jj)

            for j in end:
                result[cast_ids[i]].append(j)

            sim = decide_sim[ind].max(0)
            for j in np.argsort(1 - sim):
                result[cast_ids[i]].append(decided[j])

        for i, cast_id in enumerate(cast_ids):
            vals = result[cast_id]
            vals = unique_list(vals)
            result[cast_id] = [candi_ids[j] for j in vals]
    # Writing the result
    with open(args.submission, 'w') as f:
        for key, vals in result.items():
            f.writelines('{} {}\n'.format(key, ','.join(vals)))


if __name__ == '__main__':
    decide()
    if args.mode == 'val':
        eval(args.submission, args.gt)
