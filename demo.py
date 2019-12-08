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

def load_pkl(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
        return data

def demo(images_list):
    # print(images)
    images = ['body/val/'+image['file'] for image in images_list]
    # print(images)
    present = [image['label'] for image in images_list]
    images = [Image.open(x) for x in images]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    # max_height = max(heights)
    max_height = 400
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    # for im in images:
    #     im=ImageOps.expand(im, border=100, fill="green")
    #     print(hasattr(im, 'filename'))
    #     im.save(im.filename)
    i=0
    for im in images:
        if(present[i] == 'True'):
            border_color="green"
        else:
            border_color="red"    
        new_im.paste(ImageOps.expand(im.resize((im.size[0], max_height)), border=5, fill=border_color), (x_offset,0))
        x_offset += im.size[0]
        i+=1

    new_im.save('demo.jpg')
    new_im.show()

def eval(submission_file, gt_file):
    gt_dict = read_gt(gt_file)
    submission = parse_submission(submission_file)
    original_candidates = gt_dict.get(args.movcast)
    # print(original_candidates)
    ori_size = len(original_candidates)
    # print(len(submission.get(args.movcast)))
    if(args.movcast not in submission.keys()):
        print("Sorry!! Not processed yet!!!")
        return
    predicted_candidates = submission.get(args.movcast)[:6]
    result = load_pkl('data/retina_val.pkl')
    movie_name=args.movcast.split('_')[0]

    result_cand = result[movie_name]['candidates']
    result_cast = result[movie_name]['cast']
    candi_list=[]

    for cast in result_cast:
        # print(cast['id'])
        if(cast['id'] == args.movcast):
            candi_list.append({'file':cast['file'], 'label':'True'})
            # print('The cast is ', cast['file'])
    # i=0
    for candi in result_cand:
        # print(candi)
        if candi['id'] in predicted_candidates:
            # print(candi['file'])
           
            if(candi['id'] in original_candidates):
                candi_list.append({'file':candi['file'], 'label':'True'})
            else:
                candi_list.append({'file':candi['file'], 'label':'False'})    
            # i=i+1
            # if(i==8):
            #     break
    # print(candi_list)
    demo(candi_list)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='data/val_label.json')
    parser.add_argument('--submission', type=str, default='val_result.txt')
    parser.add_argument('--movcast', type=str, default='tt0281358_nm0314514')
    args = parser.parse_args()

    eval(args.submission, args.gt)