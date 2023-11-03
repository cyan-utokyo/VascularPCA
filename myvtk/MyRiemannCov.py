import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA, KernelPCA
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
from scipy.spatial.distance import euclidean
from myvtk.Mypca import *
import shutil
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec
from myvtk.scores import *
import csv
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal, kde
import seaborn as sns
import copy
import joblib
from myvtk.geometry import *
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from minisom import MiniSom
from sklearn.neighbors import KernelDensity
from myvtk.synthetic import *
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from myvtk.Mymetrics import *
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
import warnings
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import minimize
from myvtk.mygeodesic_plot import *
import platform
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from collections import defaultdict
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import pearsonr
import networkx as nx  # 导入NetworkX库

def draw_covariance_heatmap(data, savepath):
    """
    使用全部数据绘制协方差矩阵的热图。
    
    :param data: 形状为(m, n)的数据集，其中m是数据点的数量，n是特征的数量。
    :param savepath: 图片保存路径。
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 计算协方差矩阵
    cov_matrix = EmpiricalCovariance().fit(data_scaled).covariance_

    # 绘制热图
    plt.figure(figsize=(8, 8), dpi=300)
    sns.heatmap(cov_matrix, annot=False, cmap='viridis')
    plt.title("Covariance Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def detect_linearly_related_groups(data, sample_size, correlation_threshold=0.95):
    """
    从数据集中挑选特定数量的数据点，计算协方差矩阵，检测并在结束时输出线性近似相关的团的总结。

    :param data: 形状为(m, n)的数据集，其中m是数据点的数量，n是特征的数量。
    :param sample_size: 要从数据中随机选取的样本数量。
    :param correlation_threshold: 设定的相关系数阈值。
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    m, n = data.shape
    connected_components_summary = []

    # 随机选择指定数量的数据点
    selected_indices = np.random.choice(m, sample_size, replace=False)
    selected_data = data_scaled[selected_indices]
    
    # 创建图来表示行之间的近似线性关系
    G = nx.Graph()
    for row_i in range(n):
        G.add_node(row_i)
        for row_j in range(row_i + 1, n):
            corr, _ = pearsonr(selected_data[:, row_i], selected_data[:, row_j])
            if abs(corr) > correlation_threshold:
                G.add_edge(row_i, row_j)
    
    # 找到图中的所有连通分量（即线性近似相关的团或簇）
    connected_components = list(nx.connected_components(G))

    # 收集连通分量的信息以备最后输出
    if connected_components:
        connected_components_summary.extend(connected_components)

    # 在所有操作执行完后打印总结
    return connected_components_summary

def repeated_detection(data, sample_size, repetitions, correlation_threshold=0.95):
    """
    多次执行 detect_linearly_related_groups 函数，并对结果进行统计。

    :param data: 形状为(m, n)的数据集，其中m是数据点的数量，n是特征的数量。
    :param sample_size: 要从数据中随机选取的样本数量。
    :param repetitions: 执行 detect_linearly_related_groups 函数的次数。
    :param correlation_threshold: 设定的相关系数阈值。
    :return: 每次执行的统计结果。
    """
    overall_summary = []

    for _ in range(repetitions):
        summary = detect_linearly_related_groups(data, sample_size, correlation_threshold)
        overall_summary.extend(summary)

    # 统计结果
    results_statistics = {}
    for component in overall_summary:
        if len(component) > 2:  # 只考虑长度大于2的组件
            component = tuple(sorted(component))  # 对组件进行排序以保证一致性
            if component in results_statistics:
                results_statistics[component] += 1
            else:
                results_statistics[component] = 1

    # 只保留出现次数至少为重复次数一半的组件
    half_repetitions = repetitions // 2
    filtered_results = {k: v for k, v in results_statistics.items() if v > half_repetitions}

    # 按出现次数排序
    sorted_results = dict(sorted(filtered_results.items(), key=lambda item: item[1], reverse=True))

    return sorted_results