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
from myvtk.General import mkdir
from datetime import datetime
import geomstats.geometry.pre_shape as pre_shape
import geomstats.geometry.discrete_curves as dc
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import HypersphereMetric
from scipy.spatial import distance
from myvtk.centerline_preprocessing import *
from scipy import interpolate
import matplotlib
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, chebyshev, cityblock, cosine, correlation, sqeuclidean

def compute_dtw(srvf_curves, curves, Q2):
    warp_function_list = []
    transformed_Q1_list = []
    transformed_L1_list = []
    for idx in range(len(srvf_curves)):
        # 假设Q1和Q2是两个已经计算好的SRVF，shape都为（64,3）
        Q1 = srvf_curves[idx]
        # Step 1: 对SRVF进行动态时间规整
        distance, path = fastdtw(Q1, Q2, dist=cosine)

        # Step 2: 计算最优变换
        # path是一个包含配对索引的列表，我们可以使用它来创建变换后的曲线
        # 我们假设Q1是我们想要对齐的曲线，Q2是参考曲线
        transformed_Q1 = np.zeros(Q1.shape)
        transformed_L1 = np.zeros(Q1.shape)
        
        for pair in path:
            transformed_Q1[pair[0]] = Q2[pair[1]] # 原本是Q1[pair[1]]
            transformed_L1[pair[0]] = np.mean(curves,axis=0)[pair[1]] 
            
        # 这个transformed_Q1现在是变换后的SRVF，你可以将其转换回原始曲线空间
        # 将path转化为numpy array，以便于处理
        path_array = np.array(path)

        # 初始化warp function为一个与Q1同样长度的零向量
        warp_function = np.zeros(Q1.shape[0])

        # 根据path定义warp function
        # 对于Q1中的每一个点，warp function的值就是与之对应的Q2中的点的索引
        for i in range(path_array.shape[0]):
            warp_function[path_array[i, 0]] = path_array[i, 1]

        # 因为warp function可能不是连续的，所以我们可以通过插值来得到一个连续的warp function
        # 这里我们用线性插值
        continuous_warp_function = np.interp(np.arange(Q1.shape[0]), path_array[:,0], path_array[:,1])
        warp_function_list.append(continuous_warp_function)
        transformed_Q1_list.append(transformed_Q1)
        transformed_L1_list.append(transformed_L1)
    return np.array(warp_function_list), np.array(transformed_Q1_list), np.array(transformed_L1_list)