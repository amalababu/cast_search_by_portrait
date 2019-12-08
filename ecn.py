import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn import metrics
from argparse import ArgumentParser
from scipy.sparse import csr_matrix
from eval import eval

# Refine the result set using expanded cross neighborhood re-reanking algorithm
def ECN(features, num_queries,num_candidates, t=10,q=8):
    """
    M. Saquib Sarfraz, Arne Schumann, Andreas Eberle, Ranier Stiefelhagen,
    "A Pose Sensitive Embedding for Person Re-Identification with Exapanded Cross Neighborhood Re-Ranking", 
    https://arxiv.org/abs/1711.10378 2017. [accepted at CVPR 2018]
    """
    # Calculate the pairwise distances between the features extracted from cast and candidates, available metrics include euclidean, cosine, cityblock, l1, l2, manhattan
    r_dist=metrics.pairwise.pairwise_distances(features,features,metric='euclidean',n_jobs=-1)
    # Initial rank is obtained by sorting the distance
    initial_rank=r_dist.argsort().astype(np.int32)
    
    # Extracting the top t neighbors
    top_t_neighbor=initial_rank[:,1:t+1]
    t_indices=top_t_neighbor[num_queries:,:].T
    next_2_t_neighbor=np.transpose(initial_rank[t_indices,1:q+1],[0,2,1])
    next_2_t_neighbor=np.reshape(next_2_t_neighbor,(t*q,num_candidates))
    t_indices=np.concatenate((t_indices,next_2_t_neighbor),axis=0)    
    
    # Extracting q neighbors of each of the t neighbors of each cast
    q_indices=top_t_neighbor[:num_queries,:].T
    next_2_q_neighbor=np.transpose(initial_rank[q_indices,1:q+1],[0,2,1])
    next_2_q_neighbor=np.reshape(next_2_q_neighbor,(t*q,num_queries))
    
    q_indices=np.concatenate((q_indices,next_2_q_neighbor),axis=0)
    # Distances of each of the t neighbors from a given probe
    t_neighbor_dist=r_dist[t_indices,:num_queries]
    # Distances of each of the q neighbors of t neighbors of probe
    q_neighbor_dist=r_dist[q_indices,num_queries:]
    q_neighbor_dist=np.transpose(q_neighbor_dist,[0,2,1])
    ecn_dist=np.mean(np.concatenate((q_neighbor_dist,t_neighbor_dist),axis=0),axis=0)
    return ecn_dist.T

