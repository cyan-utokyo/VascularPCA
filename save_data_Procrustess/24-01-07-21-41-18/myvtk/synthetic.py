import numpy as np
import matplotlib.pyplot as plt
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
# import geomstats.geometry.pre_shape as pre_shape
# import geomstats.geometry.discrete_curves as dc
# from geomstats.geometry.euclidean import EuclideanMetric
# from geomstats.geometry.hypersphere import HypersphereMetric
from scipy.spatial import distance
from myvtk.centerline_preprocessing import *
from scipy import interpolate
import matplotlib
import matplotlib.cm as cm
from scipy.spatial.distance import euclidean
from myvtk.Mypca import *
# from myvtk.make_fig import *
import shutil
import os
# from myvtk.dtw import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec
from myvtk.Myscores import *
import csv
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal, kde
import seaborn as sns
import copy
import joblib
from myvtk.geometry import *
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from minisom import MiniSom
from sklearn.neighbors import KernelDensity
from myvtk.synthetic import *

def plot_recovered(recovered, curvatures, torsions, title_prefix, weights, savepath):
    """
    绘制recovered数据的曲率和扭转。
    
    :param recovered: 要绘制的数据。
    :param curvatures: 曲率的数据。
    :param torsions: 扭转的数据。
    :param title_prefix: 图的标题前缀，例如"U", "V", "C"等。
    :return: None
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    linestyles = ['-', '--', '-.', ':', 'dashdot']

    # 绘制曲率子图
    for i, rec in enumerate(recovered):
        U_c, _ = compute_curvature_and_torsion(rec)
        U_c = np.convolve(U_c, weights, 'valid')
        axes[0].plot(U_c, color='black', linestyle=linestyles[i], label=f"{title_prefix}_recovered {i+1}")
    
    # 绘制curvatures的平均值和标准差
    curv_mean = np.mean(curvatures, axis=0)
    curv_std = np.std(curvatures, axis=0)
    axes[0].errorbar(range(len(curv_mean)), curv_mean, yerr=curv_std, color='dimgray', alpha=0.5, label=f"{title_prefix}_curvatures mean ± std")

    # 绘制扭转子图
    for i, rec in enumerate(recovered):
        _, U_t = compute_curvature_and_torsion(rec)
        U_t = np.convolve(U_t, weights, 'valid')
        axes[1].plot(U_t, color='black', linestyle=linestyles[i], label=f"{title_prefix}_recovered {i+1}")

    # 绘制torsions的平均值和标准差
    tors_mean = np.mean(torsions, axis=0)
    tors_std = np.std(torsions, axis=0)
    axes[1].errorbar(range(len(tors_mean)), tors_mean, yerr=tors_std, color='dimgray', alpha=0.5, label=f"{title_prefix}_torsions mean ± std")

    # 设置子图标题和图例
    axes[0].set_title(f"{title_prefix} Curvature")
    axes[1].set_title(f"{title_prefix} Torsion")
    axes[0].set_ylim([0,1.0])
    axes[1].set_ylim([-1.2,1.2])
    for ax in axes:
        ax.legend()
        ax.grid(linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(savepath)
    #plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_recovered_stats(recovered, curvatures, torsions, title_prefix, weights, savepath, color='blue'):
    """
    绘制recovered数据的曲率和扭转的平均值和标准差。
    
    :param recovered: 要绘制的数据。
    :param curvatures: 曲率的数据。
    :param torsions: 扭转的数据。
    :param title_prefix: 图的标题前缀，例如"U", "V", "C"等。
    :return: None
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # 计算recovered的平均曲率和扭转
    recovered_curvatures = [np.convolve(compute_curvature_and_torsion(rec)[0], weights, 'valid') for rec in recovered]
    recovered_torsions = [np.convolve(compute_curvature_and_torsion(rec)[1], weights, 'valid') for rec in recovered]

    rec_curv_mean = np.mean(recovered_curvatures, axis=0)
    rec_curv_std = np.std(recovered_curvatures, axis=0)

    rec_tors_mean = np.mean(recovered_torsions, axis=0)
    rec_tors_std = np.std(recovered_torsions, axis=0)

    # 绘制recovered的平均曲率和标准差
    axes[0].errorbar(range(len(rec_curv_mean)), rec_curv_mean, yerr=rec_curv_std, color=color, label=f"{title_prefix}_recovered mean ± std")
    
    # 绘制curvatures的平均值和标准差
    curv_mean = np.mean(curvatures, axis=0)
    curv_std = np.std(curvatures, axis=0)
    axes[0].errorbar(range(len(curv_mean)), curv_mean, yerr=curv_std, color='dimgray', alpha=0.5, label=f"{title_prefix}_curvatures mean ± std")

    # 绘制recovered的平均扭转和标准差
    axes[1].errorbar(range(len(rec_tors_mean)), rec_tors_mean, yerr=rec_tors_std, color=color, label=f"{title_prefix}_recovered mean ± std")

    # 绘制torsions的平均值和标准差
    tors_mean = np.mean(torsions, axis=0)
    tors_std = np.std(torsions, axis=0)
    axes[1].errorbar(range(len(tors_mean)), tors_mean, yerr=tors_std, color='dimgray', alpha=0.5, label=f"{title_prefix}_torsions mean ± std")

    # 设置子图标题和图例
    axes[0].set_title(f"{title_prefix} Curvature")
    axes[1].set_title(f"{title_prefix} Torsion")
    axes[0].set_ylim([0,1.0])
    axes[1].set_ylim([-1.2,1.2])
    for ax in axes:
        ax.legend()
        ax.grid(linestyle=":", alpha=0.5)
        actual_ticks = np.linspace(0, 60, 5)  # 假设数据范围是0到60
        display_ticks = np.linspace(0, 1, 5)  # 希望显示的范围是0到1
        
        ax.set_xticks(actual_ticks)
        ax.set_xticklabels(display_ticks)
    
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
    #plt.show()



def plot_stats_by_label(recovered, labels, curvatures, torsions, title_prefix, weights, savepath_prefix):
    # 创建一个标签到颜色的映射
    label_to_color = {
        0: 'red',
        1: 'blue',
        2: 'orange'
    }
    unique_labels = np.unique(labels)
    for label in unique_labels:
        # 选择这个标签的数据
        selected_recovered = [recovered[i] for i in range(len(recovered)) if labels[i] == label]
        # 使用该标签的颜色
        color = label_to_color[label]
        # 使用标签来保存图片，例如：U_0_srvf_synthetic.png
        savepath = savepath_prefix + f"_{label}_srvf_synthetic.png"
        plot_recovered_stats(selected_recovered, curvatures, torsions, title_prefix, weights, savepath, color=color)