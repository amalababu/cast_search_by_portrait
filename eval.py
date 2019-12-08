import numpy
import os
import os.path as osp
import json
from random import shuffle
import argparse
from termcolor import colored
import pickle
import sys
import PIL
from PIL import Image, ImageOps


def parse_submission(submission_file):
    with open(submission_file) as f:
        lines = f.readlines()
    submission = {}
    for line in lines:
        words = line.strip().split()
        if len(words) != 2:
            print('Format Error!')
            return None
        key = words[0].strip()
        ret = words[1].strip().split(',')
        unique_ret = []
        appeared_set = set()
        for x in ret:
            if x not in appeared_set:
                unique_ret.append(x)
                appeared_set.add(x)
        submission[key] = unique_ret
    return submission


def read_gt(gt_file):
    with open(gt_file) as f:
        data = json.load(f)
    gt_dict = {}
    for key, value in data.items():
        gt_dict[key] = set(value)
    return gt_dict

def get_rank(gt_list, ret_list):
    # gt_list -> all the movie candidates
    # ret_list -> first n candidates from the result
    s1=set(gt_list)
    s2=set(ret_list)

    if s1.intersection(s2):
        return 1

    return 0    

def get_rank_n(gt_dict, ret_dict, n):
    hit = 0.0
    query_num = len(gt_dict.keys())
    # For each movie from gt_dict
    # Pass first n values from ret__dict and full gt_dict to a new function
    # the function will return 1 if the passed value is present otherwise 0
    for k,x in gt_dict.items():
        if ret_dict.get(k) is None:
            # If the movie present in val_label is not present in val_result
            query_num -=1
            continue
        hit += get_rank(x, ret_dict.get(k)[:n])

    return hit/query_num    

def get_AP(gt_set, ret_list):
    hit = 0
    AP = 0.0
    for k, x in enumerate(ret_list):
        if x in gt_set:
            hit += 1
            prec = hit / (k+1)
            AP += prec
    AP /= len(gt_set)
    return AP


def get_mAP(gt_dict, ret_dict):
    mAP = 0.0
    # min = 1
    # min_id = 1
    not_found = 0
    query_num = len(gt_dict.keys())
    s = set()
    f = set()
    for key, gt_set in gt_dict.items():
        if ret_dict.get(key) is None:
            AP = 0
            query_num = query_num -1
            not_found+=1
            movie,_ = key.split("_")
            s.add(movie)
        else:
            movie,_ = key.split("_")
            f.add(movie)
            AP = get_AP(gt_set, ret_dict[key])
            
        mAP += AP
    mAP /= query_num
   
    print("Not found movies_cast are ", not_found)
    return mAP

# Evaluate the results using mAP, rank@1, rank@5 and rank@10 metrics
def eval(submission_file, gt_file):
    gt_dict = read_gt(gt_file)
    submission = parse_submission(submission_file)
    # Get mAP result
    mAP = get_mAP(gt_dict, submission)
    # rank = get_rank_n(gt_dict, submission,6)
    print(colored('mAP: {:.2f}'.format(mAP * 100), 'red', attrs=['bold']))
    # Get rank@1, rank@5 and rank@10 results
    for n in [1,5,10]:
        rank = get_rank_n(gt_dict, submission,n)
        print(colored('Rank{}: {:.2f}'.format(n,rank * 100), 'red', attrs=['bold']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='data/val_label.json')
    parser.add_argument('--submission', type=str, default='val_result.txt')
    args = parser.parse_args()

    eval(args.submission, args.gt)

