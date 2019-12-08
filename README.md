Forked from [https://github.com/rogeryang12/cast_search_by_portrait](https://github.com/rogeryang12/cast_search_by_portrait)
# Cast Search by Portrait
We are trying to solve for identifying all instances of a target cast using the given image of the target cast and some candidate images which are the frames of a movie with person bounding boxes, as described in WIDER Face & Person Challenge 2019 - Track 3: Cast Search by Portrait

We follow a two stage model which first identifies the prospect using his facial features and combine the results with a body feature extractor module in a Convolutional Neural Network Architecture backed by ResNext-101 using triplet loss. We make use of front-line face detection and verification models, RetinaFace and ArcFace. We also considered re-ranking algorithms to further augment the results. We compared the efficiency and the performance of the state-of-the-art re-ranking algorithms, namely, k-reciprocal encoding and expanded cross neighborhood re-ranking.

## Data preparation
Candidate images from each movie is cropped using the bounding box co-ordinates given in the WIDER CSM dataset. The cropped images are placed in $BODY folder and files named `data/body_${MODE}.pkl` are created which contain id, path and label (if mode is not `test`) information of each image.

1. Download the `train.json, val.json, test.json, val_label.json`, and put them in `data/` folder.
2. Download the original data `train.tar val.tar, test.tar`, and extract them in `$ORIGIN` folder.
3. Crop the body images into `$BODY` folder and save the body information in `data/body_train.pkl, data/body_val.pkl. data/body_test.pkl`.
```
python3 crop_body.py --origin $ORIGIN --body $BODY --mode train
python3 crop_body.py --origin $ORIGIN --body $BODY --mode val
python3 crop_body.py --origin $ORIGIN --body $BODY --mode test
```

## Face detection and face features
This section detects faces and extracts facial bounding boxes, landmarks (of eyes, nose and mouth) and scores of each image and extracts facial features.

The following commands run the result on validation set.

1. `cd RetinaFace`
2. Type ``make`` to build cxx tools.
3. Download pretrained model, put them in `model/`
    - face detection: RetinaFace-R50 ([baidu cloud](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) or [dropbox](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0))
    - face recognition: LResNet100E-IR,ArcFace@ms1m-refine-v2 ([baidu cloud](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA) or [dropbox](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0))
4. Generate file lists. A file named `${MODE}_cast.txt` which contains the paths of all cast images in all movies and a list of files named `{MODE}_movieName.txt` with paths of all candidates in all movies are created in `data/`.
```
python3 datalist.py --root $BODY --mode val
```
5. Detect face for every file list in `data/`, and save the result in `results/`. Each file contains the facial bounding boxes, landmarks and scores.
```
# ${FILE}.txt is the data list in data/
python3 deploy --txt_file data/${FILE}.txt --pkl_file results/${FILE}.pkl \
               --image_root $BODY
```
For iteratively applying `deploy.py` on each file in `data/`, the following bash script can be used.
```
for f in data/.txt; do file=$(echo ${f##/}|cut -f 1 -d '.'); tfile=$file".txt"; pfile=$file".pkl"; echo $tfile; echo $pfile; python3 deploy.py --txt_file data/$tfile --pkl_file ../results/$pfile --image_root ../body; done
```
6. Merge the result into single file `../data/retina_val.pkl`. It contains the unified information of each image which includes the facial bounding boxes, landmarks, scores, path of file, id and label.
```
python3 merge.py --mode val
```
7. Select the face bboxes and crop faces from each image of cast and candidate and store in `${MODE}_img.pkl` 
```
python3 face_det.py --image_root $BODY --mode val
```
8. Extract face features and save it into `../data/face_val.pkl`.
```
python3 face_feat.py --mode val
```


## ReID model and body features
The following commands run the result on validation set, you can replace `--mode val` with `--mode test` to get test set result.

1. `cd reid`
2. Train a reid model with batch hard triplet loss. You can use `--use_val` to train the model with training set and validation set.
```
python3 triplet.py --image_root $BODY --cnn resnext101_32x8d
```

You can download our [pre-trained model](https://drive.google.com/file/d/1GD9BJViXYfLsyPA_pe5n2mdgJNefVSxL/view?usp=sharing). This model is trained with only training data.
 
3. Extract body features.
To find the ranking list based only on body features, they need to be extracted for both cast and candidates. For this purpose, make a copy of the existing `utils/datasets.py` and then rename the file `utils/datasets_features_cast_with_candidates.py` to `utils/datasets.py` before running the below command.
```
python3 triplet.py --image_root $BODY --cnn resnext101_32x8d --eval --mode val
```
4. Move body features `triplet_resnext101_32x8d_val.pkl` to `../data/` folder.


## Final result
Run `fusion.py` to get the ranking result `val_result.txt`. 
```
python3 fusion.py --mode val
```
You can use `--face_rerank` or `--body_rerank` to do rerank on face distances matrix or body distances matrix using k-reciprocal encoding respectively. These will improve the performance.
You can use `--face_rerank_ecn` or `--body_rerank_ecn` to do rerank on face distances matrix or body distances matrix using expanded cross neighborhood re-ranking algorithm respectively. This will improve the running time in comparison to k-reciprocal encoding.

## Evaluation and Demo
Run `eval.py` to see the evaluation metrics of the ranking result in comparison with the `${MODE}` labels. It displays Mean Average Precision, Rank@1, Rank@5 and Rank@10 values.

```
python3 eval.py
```

Run `demo.py` to see an example result of a cast with its top 6 ranked images by passing the parameter values `--movcast ${MOVIE_ID}_${CAST_ID}`

```
python3 demo.py --movcast ${MOVIE_ID}_${CAST_ID}
```

# Github projects

- baseline code from [cast_search_by_portrait](https://github.com/rogeryang12/cast_search_by_portrait) 
- face detection and face recognition from [insightface](https://github.com/deepinsight/insightface)
- random erasing from [reid_baseline](https://github.com/L1aoXingyu/reid_baseline)
- k-reciprocal rerank from [person-re-reranking](https://github.com/zhunzhong07/person-re-ranking)
- expanded cross neighborhood re-rank from [expanded-cross-neighborhood](https://github.com/pse-ecn/expanded-cross-neighborhood)


# Reference 

[1] Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, and Stefanos Zafeiriou. "RetinaFace: Single-stage Dense Face Localisation in the Wild." arXiv preprint arXiv:1905.00641, 2019.

[2] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. "Arcface: Additive angular margin loss for deep face recognition." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4690-4699. 2019.

[3] Alexander Hermans, Lucas Beyer, and Bastian Leibe. "In defense of the triplet loss for person re-identification." arXiv preprint arXiv:1703.07737, 2017.

[4] Guanshuo Wang, Yufeng Yuan, Xiong Chen, Jiwei Li, Xi Zhou. "Learning discriminative features with multiple granularities for person re-identification." 2018 ACM Multimedia Conference on Multimedia Conference. ACM, 2018.

[5] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. "Random erasing data augmentation." arXiv preprint arXiv:1708.04896, 2017.

[6] Zhun Zhong , Liang Zheng, Donglin Cao, and Shaozi Li. "Re-ranking person re-identification with k-reciprocal encoding." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1318-1327. 2017.

[7] M. Saquib Sarfraz, A. Schumann, A. Eberle, and R. Stiefelhagen. A   pose-sensitive   embedding   for   person   re-
identification with expanded cross neighborhood re-ranking. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 420–429, 2018.
