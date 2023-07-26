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
from scipy.spatial.distance import euclidean
from myvtk.dynamic_time_warp import *
from myvtk.General import *
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.decomposition import PCA

def PCA_training_and_test(train_curves, test_curves, n_components, standardization=1):
    print ("standarize training data and test data!!")
    pca = PCA(n_components=n_components)
    if standardization ==1:
        means = np.mean(train_curves, axis=0)
        stds = np.std(train_curves, axis=0)
        # 只对标准差非零的特征进行标准化
        non_zero_stds = stds != 0
        train_curves[:, non_zero_stds] = (train_curves[:, non_zero_stds] - means[non_zero_stds]) / stds[non_zero_stds]
        test_curves[:, non_zero_stds] = (test_curves[:, non_zero_stds] - means[non_zero_stds]) / stds[non_zero_stds]
        train_res = pca.fit_transform(train_curves)
        test_res = pca.transform(test_curves)
    else:
        train_res = pca.fit_transform(train_curves)
        test_res = pca.transform(test_curves)
    return train_res, test_res, pca

def plot_variance(pca, train_res, test_res, savepath):
    styles = [{'linestyle': '-', 'linewidth': 2},  # 实线，线宽1
          {'linestyle': '--', 'linewidth': 1},  # 虚线，线宽2
          {'linestyle': '-.', 'linewidth': 1},  # 破折线，线宽3
          {'linestyle': ':', 'linewidth': 1},  # 点线，线宽4
          {'linestyle': '-', 'linewidth': 1}]  # 实线，线宽5

    cmap = matplotlib.cm.get_cmap("RdGy")
    # 创建新的图像和 GridSpec 对象
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2])
    
    # 子图1：主成分的贡献度
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(np.cumsum(pca.explained_variance_ratio_), marker='o',c="k")
    ax0.set_ylabel('CC-R') # Cumulative contribution ratio
    ax0.grid(True)

    # 子图2：主成分的 loading pattern
    ax1 = fig.add_subplot(gs[1])
    # for i in range(5):
    #     #ax1.plot(pca.components_[i], label="PC{}".format(i+1), linestyle='-', linewidth=2)
    #     components_to_plot = pca.components_[i] if len(pca.components_[0]) <= 64 else pca.components_[i][::3]
    #     ax1.plot(components_to_plot, label="PC{}".format(i+1))
    for i, style in enumerate(styles):
        components_to_plot = pca.components_[i] if len(pca.components_[0]) <= 64 else pca.components_[i][::3]
        ax1.plot(components_to_plot, label="PC{}".format(i+1),c="k", **style)
    ax1.set_ylabel('PC')
    ax1.legend()
    ax1.grid(True)

    # 子图3：Train和Test data的各个主成分loading的分布情况，用boxplot
    ax2 = fig.add_subplot(gs[2])
    box_data = []
    labels = []
    colors = []
    for i in range(5):
        box_data.append(train_res[:, i])  # train data of ith PC
        labels.append('TPC{}'.format(i+1))
        colors.append(cmap(0.25))
        box_data.append(test_res[:, i])  # test data of ith PC
        labels.append('VPC{}'.format(i+1))
        colors.append(cmap(0.75))
        # Add a space between different PCs
        if i != 4:  # Don't add a space after the last PC
            box_data.append([])
            labels.append('')
            colors.append('none')  # 'none' means transparent
    bp = ax2.boxplot(box_data, patch_artist=True, labels=labels)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Loadings')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
    

    # train_data, test_data = data_dict["Procrustes_aligned_SRVF"]
    # train_data=train_data.reshape(train_num,-1)
    # test_data=test_data.reshape(test_num,-1)
    # train_data_std = np.std(train_data, axis=0)
    # train_data_mean = np.mean(train_data, axis=0)
    # test_data_std = np.std(test_data, axis=0)
    # test_data_mean = np.mean(test_data, axis=0)
    # pca = PCA(n_components=16)
    # train_data = zscore(train_data)
    # test_data = zscore(test_data)
    # pca.fit(train_data)
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.set_xlabel("Abscissas")
    # ax.set_ylabel("X component height")
    # ax.grid(linestyle=":", alpha=0.4)
    # for i in range(5):
    #     pca.components_[i] = pca.components_[i] * train_data_std[i] / test_data_std[i]
    #     ax.plot(pca.components_[i][::3], label="PC_{}".format(i+1))
    # plt.legend()
    # plt.savefig(bkup_dir+"ProcrustesSRVF_components.png")

class PCAHandler:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.train_res = None
        self.test_res = None
        self.pca = None

    def compute_pca(self, n_components=16):
        # PCA computation
        self.train_res, self.test_res, self.pca = PCA_training_and_test(self.train_data, self.test_data, n_components)

    def visualize_results(self, save_path):
        # Visualization
        plot_variance(self.pca, self.train_res, self.test_res, save_path)

    def compute_scores(self, train_dist, test_dist):
        # Score computation
        return get_train_test_score(self.train_res, self.test_res, train_dist, test_dist)
    
    def visualize_loadings(self, dist_dict, save_path):
        fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
        fig_count = 0
        for dist_key, dist_values in dist_dict.items():
            train_dist, test_dist = dist_values  # 取出列表中的两个值
            sc = ax[fig_count//2,fig_count%2].scatter(self.train_res[:,0],self.train_res[:,1],
                                                    c=train_dist,
                                                    cmap="turbo",
                                                    alpha=0.99,
                                                    marker="$T$")
            ax[fig_count//2,fig_count%2].scatter(self.test_res[:,0],self.test_res[:,1],
                                                c=test_dist,
                                                cmap="turbo",
                                                alpha=0.99,
                                                marker="$V$")
            # 设置子图标题的字体大小
            ax[fig_count//2,fig_count%2].set_title(dist_key, fontsize=14)
            ax[fig_count//2,fig_count%2].grid(linestyle=":", alpha=0.4)
            ax[fig_count//2,fig_count%2].set_xlabel("PC1")
            ax[fig_count//2,fig_count%2].set_ylabel("PC2")
            fig_count += 1

        fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.85)  # 修改边距
        # 在figure的右侧添加一个colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        # 设置colorbar的标签的字体大小
        cbar.set_label('Geodesic distance', fontsize=14)
        plt.savefig(save_path)
        plt.close()
