import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.stats import zscore
import glob
from myvtk.GetMakeVtk import GetMyVtk, makeVtkFile, measure_length
import pandas as pd
from scipy.spatial.transform import Rotation as R
from procrustes import orthogonal
from myvtk.General import *
from datetime import datetime
import geomstats.geometry.pre_shape as pre_shape
import geomstats.geometry.discrete_curves as dc
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import HypersphereMetric
from scipy.spatial import distance
from myvtk.centerline_preprocessing import *
from scipy import interpolate
import matplotlib
import matplotlib.cm as cm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from myvtk.customize_pca import *
from myvtk.make_fig import *
import shutil
import os
from myvtk.dtw import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec

def get_scores(res, dist):
    scores = {}
    pca_dist = []
    for j in range(res.shape[0]):
        pca_dist.append(np.linalg.norm(res[j]))
    correlation_matrix = np.corrcoef(pca_dist, dist) #dists[i])
    scores["correlation_coef"] = correlation_matrix[0,1]
    train_cosine_similarity = cosine_similarity([pca_dist, dist])[0,1]
    scores["cosine_similarity"] = train_cosine_similarity
    return scores

def get_train_test_score(train_res, test_res, train_dist, test_dist):
    train_scores = get_scores(train_res, train_dist)
    test_scores = get_scores(test_res, test_dist)
    return train_scores, test_scores

def get_pc_dist(pca_result):
    return np.linalg.norm(pca_result, axis=1)


class ScoreHandler:
    def __init__(self, data_name, dist_name, dist, pca_result,train=1):
        self.data_name = data_name
        self.dist_name = dist_name
        self.dist = dist
        pca_dist = get_pc_dist(pca_result)
        correlation = np.corrcoef(pca_dist, self.dist)[0,1] #dists[i])
        similarity = cosine_similarity([pca_dist, self.dist])[0,1]
        self.score = {"correlation": correlation, "similarity": similarity}
    
    

