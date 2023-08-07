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
# from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from myvtk.dynamic_time_warp import *
from myvtk.General import *
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde, multivariate_normal, kde, entropy
from scipy.spatial.distance import jensenshannon
import seaborn as sns


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
    
def plot_single_scatter(train_res, test_res, train_dist, test_dist, dist_key, save_path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    sc = ax.scatter(train_res[:, 0], train_res[:, 1],
                    c=train_dist, cmap="RdGy", alpha=0.99, marker="$T$")
    ax.scatter(test_res[:, 0], test_res[:, 1],
               c=test_dist, cmap="RdGy", alpha=0.99, marker="$V$")

    ax.set_title(dist_key, fontsize=14)
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # 获取子图的位置
    pos = ax.get_position()
    # 为每个子图添加colorbar，位于右上角，小型尺寸
    cbar_ax = fig.add_axes([pos.x1 - 0.05, pos.y1 - 0.7, 0.015, 0.2])

    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Geodesic distance', fontsize=8)
    cbar.ax.yaxis.set_ticks_position('left')

    plt.savefig(save_path)
    plt.close()

def plot_scatter_loadings(train_res, test_res, dist_dict, save_path_prefix):
    fig_count = 0
    for dist_key, dist_values in dist_dict.items():
        train_dist, test_dist = dist_values  # 取出列表中的两个值
        save_path = f"{save_path_prefix}_fig{fig_count}.png"
        plot_single_scatter(train_res, test_res, train_dist, test_dist, dist_key, save_path)
        fig_count += 1




class PCAHandler:
    def __init__(self, train_data, test_data, n_components=16,standardization=1):
        self.train_data = train_data
        self.test_data = test_data
        self.train_mean = np.mean(train_data,axis=0)
        self.train_std = np.std(train_data,axis=0)
        self.train_res = None
        self.pca = None
        
        self.standardization = standardization
        self.n_components = n_components
        self.zero_stds_original = None
        self.train_kde = None
        if test_data is not None:
            self.test_mean = np.mean(test_data,axis=0)
            self.test_std = np.std(test_data, axis=0)
        else:
            self.test_mean = None
            self.test_std = None
        self.test_res = None
        self.test_kde = None

    def compute_kde(self):
        train_kde = gaussian_kde(self.train_res.T)
        self.train_kde = train_kde
        if self.test_data is not None:
            test_kde = gaussian_kde(self.test_res.T)
            self.test_kde = test_kde
        # test_kde = gaussian_kde(self.test_res.T)
        
        
    
    def compute_train_test_js_divergence(self): 
        # 计算train和test的Jensen-Shannon散度
        train_kde = self.train_kde
        test_kde = self.test_kde
        # 注意，需要将概率密度函数的结果转换为概率分布
        train_prob = train_kde.pdf(self.train_res.T)
        train_prob /= train_prob.sum()
        test_prob = test_kde.pdf(self.train_res.T)
        test_prob /= test_prob.sum()
        # 计算并返回JS散度
        train_test_js_divergence = jensenshannon(train_prob, test_prob)**2  # Square the result to get the divergence
        return train_test_js_divergence

    def visualize_results(self, save_path):
        # Visualization pca的各种性质，如主成分的形状，贡献度，train和test的分布情况
        plot_variance(self.pca, self.train_res, self.test_res, save_path)
    
    def visualize_loadings(self, dist_dict, save_path):
        # Visualization results中的第一第二主成分的散点图
        plot_scatter_loadings(self.train_res, self.test_res, dist_dict, save_path)

    def inverse_transform_from_loadings(self, loadings):
        # 从loadings反推出原始数据
        standardized_data = self.pca.inverse_transform(loadings)
        if self.standardization == 0:
            return standardized_data
        else:
            return standardized_data * self.train_std + self.train_mean

    def PCA_training_and_test(self):
        pca = PCA(n_components=self.n_components)
        if self.standardization ==1:
            # self.train_data = 
            self.train_res = pca.fit_transform(zscore(self.train_data))
            if self.test_data is not None:
                # self.test_data = zscore(self.test_data)
                self.test_res = pca.transform(zscore(self.test_data))
            self.pca = pca
        else:
            self.train_res = pca.fit_transform(self.train_data)
            if self.test_data is not None:
                self.test_res = pca.transform(self.test_data)
            self.pca = pca
        
        return self.train_res, self.test_res, self.pca
    
    def plot_scatter_kde(self, savepath):
        cmap = plt.get_cmap('RdGy')

        # Create a DataFrame from the PCA results for easier plotting
        df_train = pd.DataFrame(self.train_res[:, :2], columns=["PC1", "PC2"])
        df_train['Type'] = 'Train'
        df_test = pd.DataFrame(self.test_res[:, :2], columns=["PC1", "PC2"])
        df_test['Type'] = 'Test'
        df = pd.concat([df_train, df_test])

        # Create a joint plot with a hue parameter based on the 'Type' column
        g = sns.jointplot(data=df, x='PC1', y='PC2', hue='Type', palette='RdGy',kind='kde', fill=False,joint_kws={"zorder": 0, "alpha": 1.0})
        # Add scatter plots
        g.ax_joint.scatter(df_train['PC1'], df_train['PC2'], marker="s", s=30,color=cmap(0.18),label='Train',edgecolors='white')
        g.ax_joint.scatter(df_test['PC1'], df_test['PC2'],  marker="X", s=30,color=cmap(0.82),label='Test',edgecolors='white')

        # g.plot_marginals(sns.histplot, color='dimgray')
        # Add grid
        g.ax_joint.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(savepath, dpi=300)
        plt.close()
    

