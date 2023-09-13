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
from myvtk.customize_pca import *
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


warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")

PCA_N_COMPONENTS = 16
SCALETO1 = False
PCA_STANDARDIZATION = 1

# 获取当前时间
start_time = datetime.now()
smooth_scale = 0.01
# 将时间格式化为 'yymmddhhmmss' 格式
dir_formatted_time = start_time.strftime('%y-%m-%d-%H-%M-%S')
bkup_dir = mkdir("./", "save_data_Procrustess")
bkup_dir = mkdir(bkup_dir, dir_formatted_time)
current_file_path = os.path.abspath(__file__)
current_file_name = os.path.basename(__file__)
backup_file_path = os.path.join(bkup_dir, current_file_name)
log = open(bkup_dir+"log.txt", "w")
log.write("Start at: {}\n".format(dir_formatted_time))
log.write("PCA_N_COMPONENTS:"+str(PCA_N_COMPONENTS)+"\n")
unaligned_curves = []
Files = []
radii = []
Curvatures = []
Torsions = []
Typevalues = [] 
# window size
window_size = 4
# calculate moving averages using numpy convolve
weights = np.repeat(1.0, window_size)/window_size
pre_files = glob.glob("./scaling/resamp_attr_ascii/vmtk64a/*.vtk")
shapetype = pd.read_csv("./UVCS_class.csv", header=None)
ill=pd.read_csv("./illcases.txt",header=None)
ill = np.array(ill[0])
for idx in range(len(pre_files)):
    # filename = pre_files[idx].split("\\")[-1].split(".")[0][:-8]
    filename = os.path.splitext(os.path.basename(pre_files[idx]))[0][:-8]
    # print (filename)
    if filename in ill:
        print (filename, "is found in illcases.txt, skip")
        continue
    # print (filename)
    new_type_value = shapetype.loc[shapetype[0] == filename, 2].iloc[0]
    Typevalues.append(new_type_value)
    pt, Curv, Tors, Radius, Abscissas, ptns, ftangent, fnormal, fbinormal = GetMyVtk(pre_files[idx], frenet=1)
    Files.append(pre_files[idx])
    pt = pt-np.mean(pt,axis=0)
    unaligned_curves.append(pt[::-1])
    radii.append(Radius[::-1])
    sma_curv = np.convolve(Curv, weights, 'valid')
    Curvatures.append(sma_curv[::-1])
    sma_tors = np.convolve(Tors, weights, 'valid')
    Torsions.append(sma_tors[::-1])
unaligned_curves = np.array(unaligned_curves)
geometry_dir = mkdir(bkup_dir, "geometry")
Typevalues = np.array(Typevalues)
######################################
#  这一段用来检验曲率和扭率的计算是否正确
# 曲率的计算是在compute_curvature_and_torsion函数中进行的
# for i in range(len(unaligned_curves)):
#     # 计算曲率必须在真实尺度下进行
#     fig = plt.figure(dpi=300,figsize=(10,6))
#     ax = fig.add_subplot(111)
#     ax.plot(compute_curvature_and_torsion(unaligned_curves[i])[0],color="red")
#     ax.plot(Curvatures[i],color="pink")
#     ax.plot(compute_curvature_and_torsion(unaligned_curves[i])[1],color="blue")
#     ax.plot(Torsions[i],color="cyan")
#     ax.set_title("{},Type {},MType {}".format(os.path.splitext(os.path.basename(Files[i]))[0][:-8],
#                                               shapetype.loc[shapetype[0] == os.path.splitext(os.path.basename(Files[i]))[0][:-8], 1].iloc[0],
#                                                 shapetype.loc[shapetype[0] == os.path.splitext(os.path.basename(Files[i]))[0][:-8], 2].iloc[0]))
#     plt.savefig(geometry_dir+"/{}.png".format(os.path.splitext(os.path.basename(Files[i]))[0][:-8]))
#     plt.close()
######################################

if SCALETO1:
    for i in range(len(unaligned_curves)):
        unaligned_curves[i] = unaligned_curves[i]*(1.0/measure_length(unaligned_curves[i]))
radii = np.array(radii)
Curvatures = np.array(Curvatures)
Torsions = np.array(Torsions)

# 为每个不同的字母分配一个唯一的数字
mapping = {letter: i for i, letter in enumerate(set(Typevalues))}
# 使用映射替换原始列表中的每个字母
numeric_lst = [mapping[letter] for letter in Typevalues]
# print(numeric_lst)

########################################
# plot各种type的平均曲率和扭率

C_curvatures = []
C_torsions = []
S_curvatures = []
S_torsions = []
U_curvatures = []
U_torsions = []
V_curvatures = []
V_torsions = []
for i in range(len(unaligned_curves)):
    if Typevalues[i] == "C":
        C_curvatures.append(Curvatures[i])
        C_torsions.append(Torsions[i])
    elif Typevalues[i] == "S":
        S_curvatures.append(Curvatures[i])
        S_torsions.append(Torsions[i])
    elif Typevalues[i] == "U":
        U_curvatures.append(Curvatures[i])
        U_torsions.append(Torsions[i])
    elif Typevalues[i] == "V":
        V_curvatures.append(Curvatures[i])
        V_torsions.append(Torsions[i])
C_curvatures = np.array(C_curvatures)
C_torsions = np.array(C_torsions)
U_curvatures = np.array(U_curvatures)
U_torsions = np.array(U_torsions)
V_curvatures = np.array(V_curvatures)
V_torsions = np.array(V_torsions)
S_curvatures = np.array(S_curvatures)
S_torsions = np.array(S_torsions)
print ("count CUVS: ")
print (len(C_curvatures),len(U_curvatures),len(V_curvatures),len(S_curvatures))



def setup_axes(position, ymin, ymax):
    ax = fig.add_subplot(position)
    ax.set_ylim(ymin, ymax)
    ax.grid(linestyle=":", alpha=0.5)
    ax.tick_params(axis='y', colors='k', labelsize=8)  # 设置y轴的颜色和字体大小
    ax.spines['left'].set_color('k')  # 设置y轴线的颜色
    
    # 设置x轴刻度和标签
    actual_ticks = np.linspace(0, 60, 5)  # 假设数据范围是0到60
    display_ticks = np.linspace(0, 1, 5)  # 希望显示的范围是0到1
    
    ax.set_xticks(actual_ticks)
    ax.set_xticklabels(display_ticks)
    
    return ax
def plot_with_errorbars(ax, ax2, curv_data, tors_data, line_alpha=1, errorbar_alpha=0.2):
    mean_curv = np.mean(curv_data, axis=0)
    std_curv = np.std(curv_data, axis=0)
    mean_tors = np.mean(tors_data, axis=0)
    std_tors = np.std(tors_data, axis=0)

    ax.plot(mean_curv, color="r", linewidth=1, alpha=line_alpha)
    ax.fill_between(range(len(mean_curv)), mean_curv - std_curv, mean_curv + std_curv, color="r", alpha=errorbar_alpha)
    ax2.plot(mean_tors, color="k", linestyle='--', linewidth=1, alpha=line_alpha)
    ax2.fill_between(range(len(mean_tors)), mean_tors - std_tors, mean_tors + std_tors, color="k", alpha=errorbar_alpha)

# 计算curvature统计
fig = plt.figure(dpi=300, figsize=(10, 4))
ax1 = setup_axes(221, 0, 1.2)
ax2 = setup_axes(222, 0, 1.2)
ax3 = setup_axes(223, 0, 1.2)
ax4 = setup_axes(224, 0, 1.2)
plot_with_errorbars(ax1, ax1, C_curvatures, Curvatures)
plot_with_errorbars(ax2, ax2, S_curvatures, Curvatures)
plot_with_errorbars(ax3, ax3, U_curvatures, Curvatures)
plot_with_errorbars(ax4, ax4, V_curvatures, Curvatures)
ax1.set_title("C")
ax2.set_title("S")
ax3.set_title("U")
ax4.set_title("V")
plt.tight_layout()
plt.savefig(geometry_dir + "/Curvatures_GroupVsTotal.png")
plt.close()

# 计算torsion统计
fig = plt.figure(dpi=300, figsize=(10, 4))
ax1 = setup_axes(221,-1.5, 1.5)
ax2 = setup_axes(222,-1.5, 1.5)
ax3 = setup_axes(223,-1.5, 1.5)
ax4 = setup_axes(224,-1.5, 1.5)
plot_with_errorbars(ax1, ax1, C_torsions, Torsions)
plot_with_errorbars(ax2, ax2, S_torsions, Torsions)
plot_with_errorbars(ax3, ax3, U_torsions, Torsions)
plot_with_errorbars(ax4, ax4, V_torsions, Torsions)
ax1.set_title("C")
ax2.set_title("S")
ax3.set_title("U")
ax4.set_title("V")
plt.tight_layout()
plt.savefig(geometry_dir + "/Torsions_GroupVsTotal.png")
plt.close()

############
# 绘制group内的曲率和扭率对比全体的偏离程度的散点图
fig = plt.figure(dpi=300, figsize=(9,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(U_curvatures,axis=0), color="r",marker="o", alpha=0.5, label='U')
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(V_curvatures,axis=0), color="g",marker="s", alpha=0.5, label='V')
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(C_curvatures,axis=0), color="b",marker="^", alpha=0.5, label='C')
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(S_curvatures,axis=0), color="orange",marker="*", alpha=0.5, label='S')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(U_torsions,axis=0), color="r",marker="o", alpha=0.5, label='U')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(V_torsions,axis=0), color="g",marker="s", alpha=0.5, label='V')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(C_torsions,axis=0), color="b",marker="^", alpha=0.5, label='C')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(S_torsions,axis=0), color="orange",marker="*", alpha=0.5, label='S')
# 获取ax1的x轴和y轴范围
x1_min, x1_max = ax1.get_xlim()
y1_min, y1_max = ax1.get_ylim()
# 确保对角线从左下角连接到右上角
diag_min_1 = max(x1_min, y1_min)
diag_max_1 = min(x1_max, y1_max)
ax1.plot([diag_min_1, diag_max_1], [diag_min_1, diag_max_1], linestyle=":", color="k")
# 获取ax2的x轴和y轴范围
x2_min, x2_max = ax2.get_xlim()
y2_min, y2_max = ax2.get_ylim()
# 确保对角线从左下角连接到右上角
diag_min_2 = max(x2_min, y2_min)
diag_max_2 = min(x2_max, y2_max)
ax2.plot([diag_min_2, diag_max_2], [diag_min_2, diag_max_2], linestyle=":", color="k")
ax1.legend(loc='best') # 添加图例到子图ax1
ax2.legend(loc='best') # 添加图例到子图ax2
for ax in [ax1, ax2]:
    ax.grid(linestyle=":", alpha=0.5)
plt.savefig(geometry_dir + "/group_param_compare.png")
plt.close()
####################################
# Bootstrap
log.write("Bootstrap\n")
bootstrap_sample_size = 6
log.write("sample size:{}\n".format(bootstrap_sample_size))

def bootstrap_resampling(data, num_iterations=1000, sample_size=None):
    """为给定的数据生成Bootstrap样本"""
    n, m = data.shape  # 注意这里获取的是两个维度的大小
    if sample_size is None:
        sample_size = n  # 如果没有指定抽样大小，就使用原始数据的大小
    print ("sample_size:", sample_size)
    bootstrap_samples = []
    for _ in range(num_iterations):
        # 使用randint而不是choice，并确保按行采样
        indices = np.random.randint(0, n, size=sample_size)
        sample = data[indices, :]
        bootstrap_samples.append(sample)
    return np.array(bootstrap_samples)

# Bootstrap重采样
bootstrap_samples_U_curvature = bootstrap_resampling(U_curvatures,sample_size=bootstrap_sample_size)
bootstrap_samples_V_curvature = bootstrap_resampling(V_curvatures,sample_size=bootstrap_sample_size)
bootstrap_samples_C_curvature = bootstrap_resampling(C_curvatures,sample_size=bootstrap_sample_size)
bootstrap_samples_S_curvature = bootstrap_resampling(S_curvatures,sample_size=bootstrap_sample_size)
bootstrap_samples_overall_curvature = bootstrap_resampling(Curvatures,sample_size=bootstrap_sample_size)
bootstrap_samples_U_torsion = bootstrap_resampling(U_torsions,sample_size=bootstrap_sample_size)
bootstrap_samples_V_torsion = bootstrap_resampling(V_torsions,sample_size=bootstrap_sample_size)
bootstrap_samples_C_torsion = bootstrap_resampling(C_torsions,sample_size=bootstrap_sample_size)
bootstrap_samples_S_torsion = bootstrap_resampling(S_torsions,sample_size=bootstrap_sample_size)
bootstrap_samples_overall_torsion = bootstrap_resampling(Torsions,sample_size=bootstrap_sample_size)
print ("bootstrap_samples_U_curvature shape:", bootstrap_samples_U_curvature.shape)
def compute_bootstrap_statistics(bootstrap_samples):
    """计算Bootstrap样本的均值和标准差"""
    means = np.mean(bootstrap_samples, axis=0)
    stds = np.std(bootstrap_samples, axis=0)
    return means, stds

# 计算Bootstrap样本的统计数据
means_overall_curvature, stds_overall_curvature = compute_bootstrap_statistics(bootstrap_samples_overall_curvature)
means_overall_torsion, stds_overall_torsion = compute_bootstrap_statistics(bootstrap_samples_overall_torsion)

fig, axes = plt.subplots(4, 2, dpi=300, figsize=(10, 8))
def plot_with_shade(ax, data_samples,  title, ymin, ymax):
    if len(data_samples.shape) == 3:  # bootstrap samples
        x = range(data_samples.shape[2])
    else:  # means or stds
        x = range(data_samples.shape[0])
    # x = range(data_samples.shape[2])
    # 定义箱型图样式参数
    box_properties = {
        'color': 'dimgray',
        'linewidth': 1
    }
    whisker_properties = {
        'color': 'dimgray',
        'linewidth': 1
    }
    cap_properties = {
        'color': 'dimgray',
        'linewidth': 1
    }
    median_properties = {
        'color': 'red',
        'linewidth': 1.5
    }

    bp = ax.boxplot(data_samples, 
            showfliers=False,
            boxprops=box_properties, 
            medianprops=median_properties,
            whiskerprops=whisker_properties,
            capprops=cap_properties)
    #means = np.mean(data_samples, axis=(0, 1))
    #ax.scatter(range(1, len(means)+1), means, marker='_', color='blue', zorder=5, s=5)

    ax.set_title(title)
    ax.set_ylim(ymin, ymax)
    # 设置x轴的标签
    x_ticks = np.linspace(0, data_samples.shape[1]-1, 6)  # 6个刻度点
    x_labels = [f"{val:.1f}" for val in np.linspace(0, 1, 6)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.grid(linestyle=":", alpha=0.5)

print ("compute_bootstrap_statistics(bootstrap_samples_C_curvature)[0] shape:", compute_bootstrap_statistics(bootstrap_samples_C_curvature)[0].shape)

plot_with_shade(axes[0, 0], compute_bootstrap_statistics(bootstrap_samples_C_curvature)[0], "C - Means", 0, 0.6)
plot_with_shade(axes[0, 1], compute_bootstrap_statistics(bootstrap_samples_C_curvature)[1], "C - Stds", 0, 0.4)
plot_with_shade(axes[1, 0], compute_bootstrap_statistics(bootstrap_samples_S_curvature)[0], "S - Means", 0, 0.6)
plot_with_shade(axes[1, 1], compute_bootstrap_statistics(bootstrap_samples_S_curvature)[1], "S - Stds", 0, 0.4)
plot_with_shade(axes[2, 0], compute_bootstrap_statistics(bootstrap_samples_U_curvature)[0], "U - Means", 0, 0.6)
plot_with_shade(axes[2, 1], compute_bootstrap_statistics(bootstrap_samples_U_curvature)[1], "U - Stds", 0, 0.4)
plot_with_shade(axes[3, 0], compute_bootstrap_statistics(bootstrap_samples_V_curvature)[0], "V - Means", 0, 0.6)
plot_with_shade(axes[3, 1], compute_bootstrap_statistics(bootstrap_samples_V_curvature)[1], "V - Stds", 0, 0.4)


plt.tight_layout()
plt.savefig(geometry_dir+"Bootstrap_Distributions_with_Global_Curvature.png")
plt.close()


fig, axes = plt.subplots(4, 2, dpi=300, figsize=(10, 8))
plot_with_shade(axes[0, 0], compute_bootstrap_statistics(bootstrap_samples_C_torsion)[0], "C - Means", -0.6, 0.6)
plot_with_shade(axes[0, 1], compute_bootstrap_statistics(bootstrap_samples_C_torsion)[1], "C - Stds", 0, 1.2)
plot_with_shade(axes[1, 0], compute_bootstrap_statistics(bootstrap_samples_S_torsion)[0], "S - Means", -0.6, 0.6)
plot_with_shade(axes[1, 1], compute_bootstrap_statistics(bootstrap_samples_S_torsion)[1], "S - Stds", 0, 1.2)
plot_with_shade(axes[2, 0], compute_bootstrap_statistics(bootstrap_samples_U_torsion)[0], "U - Means", -0.6, 0.6)
plot_with_shade(axes[2, 1], compute_bootstrap_statistics(bootstrap_samples_U_torsion)[1], "U - Stds", 0, 1.2)
plot_with_shade(axes[3, 0], compute_bootstrap_statistics(bootstrap_samples_V_torsion)[0], "V - Means", -0.6, 0.6)
plot_with_shade(axes[3, 1], compute_bootstrap_statistics(bootstrap_samples_V_torsion)[1], "V - Stds", 0, 1.2)
plt.tight_layout()
plt.savefig(geometry_dir+"Bootstrap_Distributions_with_Global_Torsion.png")
plt.close()


########################################

print ("全データ（{}）を読み込みました。".format(len(pre_files)))
print ("使用できるデータ：", len(Files))
for i in range(len(Files)):
    if "BH0017_R" in Files[i]:
        base_id = i
print ("base_id:{},casename:{}で方向調整する".format(base_id, Files[base_id]))

##################################################
#  从这里开始是对齐。                             #
#  To-Do: 需要保存Procrustes对齐后的              #
#  曲线各一条，作为后续的曲线对齐的基准。           #
##################################################

a_curves = align_icp(unaligned_curves, base_id=base_id)
print ("First alignment done.")
Procrustes_curves = align_procrustes(a_curves,base_id=base_id)
print ("procrustes alignment done.")
# for i in range(len(Procrustes_curves)):
#     print ("length:", measure_length(Procrustes_curves[i]))
parametrized_curves = np.zeros_like(Procrustes_curves)
# aligned_curves = np.zeros_like(interpolated_curves)
for i in range(len(Procrustes_curves)):
    parametrized_curves[i] = arc_length_parametrize(Procrustes_curves[i])
Procrustes_curves = np.array(Procrustes_curves)

print (Procrustes_curves.shape)

# if SCALETO1:
#     # 需要把长度还原到原始曲线或1
#     for i in range(len(Procrustes_curves)):
#         aligned_length = measure_length(Procrustes_curves[i])
#         procrustes_length = measure_length(Procrustes_curves[i])
#         Procrustes_curves[i] = Procrustes_curves[i] * (1.0/procrustes_length) # 这里是把长度还原到1
# log.write("Scaled all curves to one.\n")

# shape of Procrustes_curves: (87, 64, 3)
# shape of Curvatures: (87, 61)
# interpolate Curvatures to (87, 64)


import numpy as np
from scipy.optimize import minimize

# 给定landmark_w，计算加权的Procrustes_curves

def compute_weighted_procrustes(Procrustes_curves, Curvatures, landmark_w):
    interpolated_curvatures = np.zeros((len(Curvatures), len(Procrustes_curves[0])))
    multiplicative_factors = np.zeros_like(Procrustes_curves)

    for i in range(len(Curvatures)):
        interpolated_curvatures[i] = np.interp(np.linspace(0, 1, len(Procrustes_curves[i])),
                                               np.linspace(0, 1, len(Curvatures[i])),
                                               Curvatures[i])
        multiplicative_factors[i] = Procrustes_curves[i] * interpolated_curvatures[i][:, np.newaxis]

    weighted_procrustes_curves = multiplicative_factors * landmark_w[:, np.newaxis]
    return weighted_procrustes_curves

# 根据weighted_procrustes_curves计算距离矩阵
def compute_geod_dist_mat(weighted_procrustes_curves):
    geod_dist_mat = np.zeros((len(weighted_procrustes_curves), len(weighted_procrustes_curves)))
    for i in range(len(weighted_procrustes_curves)):
        for j in range(len(weighted_procrustes_curves)):
            geodesic_d = compute_geodesic_dist_between_two_curves(weighted_procrustes_curves[i],
                                                                  weighted_procrustes_curves[j])
            geod_dist_mat[i, j] = geodesic_d
    return geod_dist_mat

# 优化的目标函数
def objective(landmark_w, Procrustes_curves, Curvatures, numeric_lst):
    # print ("numeric_lst:", numeric_lst)
    weighted_curves = compute_weighted_procrustes(Procrustes_curves, Curvatures, landmark_w)
    geod_dist_mat = compute_geod_dist_mat(weighted_curves)

    # Check for NaN values in geod_dist_mat
    # if np.isnan(geod_dist_mat).any():
    #     return 1e10  # Return a large positive value

    # PCA处理
    geod_dist_pca = PCAHandler(geod_dist_mat, None, PCA_N_COMPONENTS, PCA_STANDARDIZATION)
    reduced_data,_,_ = geod_dist_pca.PCA_training_and_test()

    # 计算四个类别的中心点
    class_centers = []
    for label in [0, 1, 2, 3]:
        # indices = np.where(numeric_lst == label)[0]
        indices = np.where(np.array(numeric_lst) == label)[0]
        # print ("label:",label,"indices:",indices)
        class_center = np.mean(reduced_data[indices], axis=0)
        class_centers.append(class_center)
    
    # 计算类别之间的距离
    distances = []
    for i in range(4):
        for j in range(i+1, 4):
            distances.append(np.linalg.norm(class_centers[i] - class_centers[j]))
    
    # 因为我们使用的是minimize函数，所以需要返回负的距离以实现最大化效果
    return -np.sum(distances)

# 使用scipy.optimize.minimize进行优化
# initial_landmark_w = np.ones_like(Procrustes_curves[0,:,0])
# res = minimize(objective, initial_landmark_w, 
#                args=(Procrustes_curves, Curvatures, numeric_lst), 
#                method='L-BFGS-B')

# optimized_landmark_w = res.x
# print(optimized_landmark_w)
initial_landmark_w = np.ones_like(Procrustes_curves[0,:,0])
random_vecs = []
random_dists = []
from tqdm import tqdm
for i in tqdm(range(1000)):
    random_vector = np.random.rand(*initial_landmark_w.shape)
    random_dist = objective(random_vector, Procrustes_curves, Curvatures, numeric_lst)
    random_vecs.append(random_vector)
    random_dists.append(random_dist)
fig = plt.figure(dpi=300, figsize=(6, 4))
ax1 = fig.add_subplot(111)
ax1.hist(random_dists, bins=20, edgecolor='black', alpha=0.7)  # 可以修改bins的值来调整柱状的宽度
ax1.set_xlabel('Distance')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of random_dists')
plt.tight_layout()
plt.savefig(bkup_dir+"random_dists_histogram.png")
plt.close()

# 获取最小值的索引
min_index = np.argmin(random_dists)

# 使用该索引获取对应的vector
optimized_landmark_w = random_vecs[min_index]


landmark_w = optimized_landmark_w
interpolated_curvatures = np.zeros((len(Curvatures), len(Procrustes_curves[0])))
weighted_procrustes_curves = np.zeros_like(Procrustes_curves)
for i in range(len(Curvatures)):
    interpolated_curvatures[i] = np.interp(np.linspace(0, 1, len(Procrustes_curves[i])), np.linspace(0, 1, len(Curvatures[i])), Curvatures[i])
for i in range(len(interpolated_curvatures)):
    for j in range(len(interpolated_curvatures[i])):
        weighted_procrustes_curves[i][j] = Procrustes_curves[i][j] * interpolated_curvatures[i][j] * landmark_w[j]

geod_dist_mat = np.zeros((len(weighted_procrustes_curves), len(weighted_procrustes_curves)))
for i in range(len(weighted_procrustes_curves)):
    for j in range(len(weighted_procrustes_curves)):
        geodesic_d = compute_geodesic_dist_between_two_curves(weighted_procrustes_curves[i], weighted_procrustes_curves[j])
        geod_dist_mat[i, j] = geodesic_d

geod_dist_pca = PCAHandler(geod_dist_mat, None, PCA_N_COMPONENTS, PCA_STANDARDIZATION)
geod_dist_pca.PCA_training_and_test()
fig = plt.figure(dpi=300, figsize=(6, 4))
ax1 = fig.add_subplot(111)
ax1.scatter(geod_dist_pca.train_res[:, 0], geod_dist_pca.train_res[:, 1], c=numeric_lst, cmap="turbo")
plt.savefig(bkup_dir + "geod_dist_pca.png")
plt.close()


# SRVF计算
Procs_srvf_curves = np.zeros_like(Procrustes_curves)
for i in range(len(Procrustes_curves)):
    Procs_srvf_curves[i] = calculate_srvf(Procrustes_curves[i])

makeVtkFile(bkup_dir+"mean_curve.vtk", np.mean(Procrustes_curves,axis=0),[],[] )
mean_srvf_inverse = inverse_srvf(np.mean(Procs_srvf_curves,axis=0),np.zeros(3))
makeVtkFile(bkup_dir+"mean_srvf.vtk", mean_srvf_inverse,[],[] )

C_curvatures_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(C_curvatures)
U_curvatures_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(U_curvatures)
S_curvatures_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(S_curvatures)
V_curvatures_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(V_curvatures)
C_torsions_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(C_torsions)
U_torsions_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(U_torsions)
S_torsions_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(S_torsions)
V_torsions_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(V_torsions)

# frechet_mean_srvf = compute_frechet_mean(Procs_srvf_curves)
# frechet_mean_srvf = frechet_mean_srvf / measure_length(frechet_mean_srvf)
# 保存数据

PCA_weight = np.mean(Curvatures, axis=0)


all_srvf_pca = PCAHandler(Procs_srvf_curves.reshape(len(Procs_srvf_curves),-1), None, PCA_N_COMPONENTS, PCA_STANDARDIZATION)
all_srvf_pca.PCA_training_and_test()
all_srvf_pca.compute_kde()
joblib.dump(all_srvf_pca.pca, bkup_dir + 'srvf_pca_model.pkl')
np.save(bkup_dir+"pca_model_filename.npy",Files )
all_pca = PCAHandler(Procrustes_curves.reshape(len(Procrustes_curves),-1), None, PCA_N_COMPONENTS, PCA_STANDARDIZATION)
all_pca.PCA_training_and_test()
all_pca.compute_kde()
joblib.dump(all_pca.pca, bkup_dir + 'pca_model.pkl')
np.save(bkup_dir+"not_std_curves.npy", all_pca.train_data)
np.save(bkup_dir+"not_std_srvf.npy", all_srvf_pca.train_data)
np.save(bkup_dir+"not_std_filenames.npy",Files)

pca_anlysis_dir = mkdir(bkup_dir, "pca_analysis")

pca_components_figname = pca_anlysis_dir+"pca_plot_variance.png"
all_pca.visualize_results(pca_components_figname)
srvf_pca_components_figname = pca_anlysis_dir + "srvf_pca_plot_variance.png"
all_srvf_pca.visualize_results(srvf_pca_components_figname)
# Define a helper function to fit KernelDensity
def fit_kde(data):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(data)
    return kde
# Extracting data based on type
C_srvf_data = all_srvf_pca.train_res[Typevalues == 'C']
U_srvf_data = all_srvf_pca.train_res[Typevalues == 'U']
S_srvf_data = all_srvf_pca.train_res[Typevalues == 'S']
V_srvf_data = all_srvf_pca.train_res[Typevalues == 'V']
# Compute KDEs using sklearn's KernelDensity
C_srvf_kde = fit_kde(C_srvf_data)
U_srvf_kde = fit_kde(U_srvf_data)
S_srvf_kde = fit_kde(S_srvf_data)
V_srvf_kde = fit_kde(V_srvf_data)
sample_num = 1000
U_synthetic = U_srvf_kde.sample(sample_num)
V_synthetic = V_srvf_kde.sample(sample_num)
C_synthetic = C_srvf_kde.sample(sample_num)
S_synthetic = S_srvf_kde.sample(sample_num)
# 定义一个函数来计算elbow值
def compute_elbow(data, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    # 计算每个点和其前后点的斜率差值
    slopes = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
    slopes_diff = [slopes[i] - slopes[i+1] for i in range(len(slopes)-1)]
    # 拐点是斜率变化最大的地方+2（因为数组从0开始并且我们要加上后一个点）
    elbow = slopes_diff.index(max(slopes_diff)) + 2
    return wcss, elbow
U_wcss, U_elbow = compute_elbow(U_synthetic)
V_wcss, V_elbow = compute_elbow(V_synthetic)
C_wcss, C_elbow = compute_elbow(C_synthetic)
S_wcss, S_elbow = compute_elbow(S_synthetic)
# 绘制elbow图

plt.figure()
plt.plot(range(1, 11), U_wcss, marker='o', label='U_synthetic')
plt.plot(range(1, 11), V_wcss, marker='o', label='V_synthetic')
plt.plot(range(1, 11), C_wcss, marker='o', label='C_synthetic')
plt.plot(range(1, 11), S_wcss, marker='o', label='S_synthetic')
# 标记每个elbow点
plt.scatter(U_elbow, U_wcss[U_elbow-1], s=150,  marker='*', label="Elbow U")
plt.scatter(V_elbow, V_wcss[V_elbow-1], s=150, marker='*', label="Elbow V")
plt.scatter(C_elbow, C_wcss[C_elbow-1], s=150,  marker='*', label="Elbow C")
plt.scatter(S_elbow, S_wcss[S_elbow-1], s=150,  marker='*', label="Elbow S")
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend()
plt.savefig(pca_anlysis_dir + "ELBOW_srvf_pca_synthetic.png")
plt.close()

# 用最佳的k值进行聚类
kmeans_U = KMeans(n_clusters=U_elbow, n_init=10).fit(U_synthetic)
kmeans_V = KMeans(n_clusters=V_elbow, n_init=10).fit(V_synthetic)
kmeans_C = KMeans(n_clusters=C_elbow, n_init=10).fit(C_synthetic)
kmeans_S = KMeans(n_clusters=S_elbow, n_init=10).fit(S_synthetic)
labels_U = kmeans_U.labels_
labels_V = kmeans_V.labels_
labels_C = kmeans_C.labels_
labels_S = kmeans_S.labels_
def generate_palette(color_base, n_colors=5):
    # 使用seaborn生成单色调色板，然后将颜色从RGB转换为十六进制格式
    return [sns.desaturate(color, 0.8) for color in sns.color_palette(color_base, n_colors=n_colors)]

synthetic_cluster_colors = {
    "U": generate_palette("Reds_r",U_elbow),  # 5 shades of red for U
    "V": generate_palette("Blues_r",V_elbow),  # 5 shades of blue for V
    "C": generate_palette("Greens_r",C_elbow),  # 5 shades of green for C
    "S": generate_palette("YlOrBr_r",S_elbow)   # 5 shades of yellow-orange-brown for S
}

def plot_synthetic_data(ax, all_data, synthetic_data, kmeans_labels, color_map, label):
    ax.scatter(all_data[:, 0], all_data[:, 1], c='dimgray', s=50, marker="x")
    for cluster_num in np.unique(kmeans_labels):
        mask = kmeans_labels == cluster_num
        ax.scatter(synthetic_data[mask, 0], synthetic_data[mask, 1], color=color_map[label][cluster_num], s=50, marker=f"${label}$")
    ax.set_title(f'{label}_synthetic')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
plot_synthetic_data(axs[0, 0], all_srvf_pca.train_res, U_synthetic, kmeans_U.labels_, synthetic_cluster_colors, 'U')
plot_synthetic_data(axs[0, 1], all_srvf_pca.train_res, V_synthetic, kmeans_V.labels_, synthetic_cluster_colors, 'V')
plot_synthetic_data(axs[1, 0], all_srvf_pca.train_res, C_synthetic, kmeans_C.labels_, synthetic_cluster_colors, 'C')
plot_synthetic_data(axs[1, 1], all_srvf_pca.train_res, S_synthetic, kmeans_S.labels_, synthetic_cluster_colors, 'S')
plt.tight_layout()
plt.savefig(pca_anlysis_dir + "srvf_pca_synthetic_scatter.png")
plt.close()

##############################
# 绘制合成曲线的curvature和torsion
U_synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(U_synthetic).reshape(sample_num, -1, 3)
U_recovered = recovered_curves(U_synthetic_inverse, True)
V_synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(V_synthetic).reshape(sample_num, -1, 3)
V_recovered = recovered_curves(V_synthetic_inverse, True)
C_synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(C_synthetic).reshape(sample_num, -1, 3)
C_recovered = recovered_curves(C_synthetic_inverse, True)
S_synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(S_synthetic).reshape(sample_num, -1, 3)
S_recovered = recovered_curves(S_synthetic_inverse, True)
# 使用函数绘制U, V, C等数据
if sample_num < 6:
    plot_recovered(U_recovered, U_curvatures, U_torsions, "U", weights, geometry_dir + "U_srvf_synthetic.png")
    plot_recovered(V_recovered, V_curvatures, V_torsions, "V", weights, geometry_dir + "V_srvf_synthetic.png")
    plot_recovered(C_recovered, C_curvatures, C_torsions, "C", weights, geometry_dir + "C_srvf_synthetic.png")
    plot_recovered(S_recovered, S_curvatures, S_torsions, "S", weights, geometry_dir + "S_srvf_synthetic.png")
else:
    # 现在，为每一个类别调用新函数
    # plot_stats_by_label(U_recovered, labels_U, U_curvatures, U_torsions, "U", weights, geometry_dir + "U")
    # plot_stats_by_label(V_recovered, labels_V, V_curvatures, V_torsions, "V", weights, geometry_dir + "V")
    # plot_stats_by_label(C_recovered, labels_C, C_curvatures, C_torsions, "C", weights, geometry_dir + "C")
    # plot_stats_by_label(S_recovered, labels_S, S_curvatures, S_torsions, "S", weights, geometry_dir + "S")
    plot_recovered_stats(U_recovered, U_curvatures, U_torsions, "U", weights, geometry_dir + "U_srvf_synthetic.png")
    plot_recovered_stats(V_recovered, V_curvatures, V_torsions, "V", weights, geometry_dir + "V_srvf_synthetic.png")
    plot_recovered_stats(C_recovered, C_curvatures, C_torsions, "C", weights, geometry_dir + "C_srvf_synthetic.png")
    plot_recovered_stats(S_recovered, S_curvatures, S_torsions, "S", weights, geometry_dir + "C_srvf_synthetic.png")
    # 调用上述函数为每个数据集绘制图形

srvf_synthetics = np.concatenate([U_synthetic, V_synthetic, C_synthetic, S_synthetic], axis=0)
print ("srvf_synthetics.shape: ", srvf_synthetics.shape)
srvf_recovers = np.concatenate([U_recovered, V_recovered, C_recovered, S_recovered], axis=0)
print ("srvf_recovers.shape: ", srvf_recovers.shape)

def compute_synthetic_curvature_and_torsion(C_recovered):
    C_srvf_synthetic_curvature = []
    C_srvf_synthetic_torsion = []
    for i in range(len(C_synthetic_inverse)):
        C_srvf_synthetic_curvature.append(np.convolve(compute_curvature_and_torsion(C_recovered[i])[0], weights, 'valid'))
        C_srvf_synthetic_torsion.append(np.convolve(compute_curvature_and_torsion(C_recovered[i])[1], weights, 'valid'))
    C_srvf_synthetic_curvature = np.array(C_srvf_synthetic_curvature)
    C_srvf_synthetic_torsion = np.array(C_srvf_synthetic_torsion)
    return C_srvf_synthetic_curvature, C_srvf_synthetic_torsion

C_srvf_synthetic_curvatures, C_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(C_recovered)
U_srvf_synthetic_curvatures, U_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(U_recovered)
V_srvf_synthetic_curvatures, V_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(V_recovered)
S_srvf_synthetic_curvatures, S_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(S_recovered)

############
# 绘制group内的曲率和扭率对比全体的偏离程度的散点图

fig = plt.figure(dpi=300, figsize=(9,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(U_srvf_synthetic_curvatures,axis=0), color="r",marker="o", alpha=0.5, label='U')
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(V_srvf_synthetic_curvatures,axis=0), color="g",marker="s", alpha=0.5, label='V')
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(C_srvf_synthetic_curvatures,axis=0), color="b",marker="^", alpha=0.5, label='C')
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(S_srvf_synthetic_curvatures,axis=0), color="orange",marker="*", alpha=0.5, label='S')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(U_srvf_synthetic_torsions,axis=0), color="r",marker="o", alpha=0.5, label='U')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(V_srvf_synthetic_torsions,axis=0), color="g",marker="s", alpha=0.5, label='V')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(C_srvf_synthetic_torsions,axis=0), color="b",marker="^", alpha=0.5, label='C')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(S_srvf_synthetic_torsions,axis=0), color="orange",marker="*", alpha=0.5, label='S')
# 获取ax1的x轴和y轴范围
x1_min, x1_max = ax1.get_xlim()
y1_min, y1_max = ax1.get_ylim()
# 确保对角线从左下角连接到右上角
diag_min_1 = max(x1_min, y1_min)
diag_max_1 = min(x1_max, y1_max)
ax1.plot([diag_min_1, diag_max_1], [diag_min_1, diag_max_1], linestyle=":", color="k")
# 获取ax2的x轴和y轴范围
x2_min, x2_max = ax2.get_xlim()
y2_min, y2_max = ax2.get_ylim()
# 确保对角线从左下角连接到右上角
diag_min_2 = max(x2_min, y2_min)
diag_max_2 = min(x2_max, y2_max)
ax2.plot([diag_min_2, diag_max_2], [diag_min_2, diag_max_2], linestyle=":", color="k")
ax1.legend(loc='best') # 添加图例到子图ax1
ax2.legend(loc='best') # 添加图例到子图ax2
for ax in [ax1, ax2]:
    ax.grid(linestyle=":", alpha=0.5)
plt.savefig(geometry_dir + "/group_param_compare_srvfSynthetic.png")
plt.close()


# 绘制合成曲线的curvature和torsion
##############################

# Computing the KDE matrix
# score = np.exp(s_kde.score_samples(data[0].reshape(1, -1)))
# Extracting and computing KDEs for all_pca data
C_data = all_pca.train_res[Typevalues == 'C']
U_data = all_pca.train_res[Typevalues == 'U']
S_data = all_pca.train_res[Typevalues == 'S']
V_data = all_pca.train_res[Typevalues == 'V']
C_kde = fit_kde(C_data)
U_kde = fit_kde(U_data)
S_kde = fit_kde(S_data)
V_kde = fit_kde(V_data)
sample_num = 1000
U_synthetic = U_kde.sample(sample_num)
V_synthetic = V_kde.sample(sample_num)
C_synthetic = C_kde.sample(sample_num)
S_synthetic = S_kde.sample(sample_num)

# 对于每个X_synthetic计算elbow值
U_wcss, U_elbow = compute_elbow(U_synthetic)
V_wcss, V_elbow = compute_elbow(V_synthetic)
C_wcss, C_elbow = compute_elbow(C_synthetic)
S_wcss, S_elbow = compute_elbow(S_synthetic)
# 绘制elbow图
plt.figure()
plt.plot(range(1, 11), U_wcss, marker='o', label='U_synthetic')
plt.plot(range(1, 11), V_wcss, marker='o', label='V_synthetic')
plt.plot(range(1, 11), C_wcss, marker='o', label='C_synthetic')
plt.plot(range(1, 11), S_wcss, marker='o', label='S_synthetic')
# 标记每个elbow点
plt.scatter(U_elbow, U_wcss[U_elbow-1], s=150,  marker='*', label="Elbow U")
plt.scatter(V_elbow, V_wcss[V_elbow-1], s=150, marker='*', label="Elbow V")
plt.scatter(C_elbow, C_wcss[C_elbow-1], s=150,  marker='*', label="Elbow C")
plt.scatter(S_elbow, S_wcss[S_elbow-1], s=150,  marker='*', label="Elbow S")
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend()
plt.savefig(pca_anlysis_dir + "ELBOW_pca_synthetic.png")
plt.close()

synthetic_cluster_colors = {
    "U": generate_palette("Reds_r",U_elbow),  # 5 shades of red for U
    "V": generate_palette("Blues_r",V_elbow),  # 5 shades of blue for V
    "C": generate_palette("Greens_r",C_elbow),  # 5 shades of green for C
    "S": generate_palette("YlOrBr_r",S_elbow)   # 5 shades of yellow-orange-brown for S
}

kmeans_U = KMeans(n_clusters=U_elbow, n_init=10).fit(U_synthetic)
kmeans_V = KMeans(n_clusters=V_elbow, n_init=10).fit(V_synthetic)
kmeans_C = KMeans(n_clusters=C_elbow, n_init=10).fit(C_synthetic)
kmeans_S = KMeans(n_clusters=S_elbow, n_init=10).fit(S_synthetic)

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

plot_synthetic_data(axs[0, 0], all_pca.train_res, U_synthetic, kmeans_U.labels_, synthetic_cluster_colors, 'U')
plot_synthetic_data(axs[0, 1], all_pca.train_res, V_synthetic, kmeans_V.labels_, synthetic_cluster_colors, 'V')
plot_synthetic_data(axs[1, 0], all_pca.train_res, C_synthetic, kmeans_C.labels_, synthetic_cluster_colors, 'C')
plot_synthetic_data(axs[1, 1], all_pca.train_res, S_synthetic, kmeans_S.labels_, synthetic_cluster_colors, 'S')

plt.tight_layout()
plt.savefig(pca_anlysis_dir + "pca_synthetic_scatter.png")
plt.close()

##############################
# 绘制合成曲线的curvature和torsion
U_synthetic_inverse = all_pca.inverse_transform_from_loadings(U_synthetic).reshape(sample_num, -1, 3)
U_recovered = recovered_curves(U_synthetic_inverse, False)
V_synthetic_inverse = all_pca.inverse_transform_from_loadings(V_synthetic).reshape(sample_num, -1, 3)
V_recovered = recovered_curves(V_synthetic_inverse, False)
C_synthetic_inverse = all_pca.inverse_transform_from_loadings(C_synthetic).reshape(sample_num, -1, 3)
C_recovered = recovered_curves(C_synthetic_inverse, False)
S_synthetic_inverse = all_pca.inverse_transform_from_loadings(S_synthetic).reshape(sample_num, -1, 3)
S_recovered = recovered_curves(S_synthetic_inverse, False)
# 使用函数绘制U, V, C等数据
if sample_num < 6:
    plot_recovered(U_recovered, U_curvatures, U_torsions, "U", weights, geometry_dir + "U_synthetic.png")
    plot_recovered(V_recovered, V_curvatures, V_torsions, "V", weights, geometry_dir + "V_synthetic.png")
    plot_recovered(C_recovered, C_curvatures, C_torsions, "C", weights, geometry_dir + "C_synthetic.png")
    plot_recovered(S_recovered, S_curvatures, S_torsions, "S", weights, geometry_dir + "S_synthetic.png")
else:
    plot_recovered_stats(U_recovered, U_curvatures, U_torsions, "U", weights, geometry_dir + "U_synthetic.png")
    plot_recovered_stats(V_recovered, V_curvatures, V_torsions, "V", weights, geometry_dir + "V_synthetic.png")
    plot_recovered_stats(C_recovered, C_curvatures, C_torsions, "C", weights, geometry_dir + "C_synthetic.png")
    plot_recovered_stats(S_recovered, S_curvatures, S_torsions, "S", weights, geometry_dir + "S_synthetic.png")

non_srvf_synthetics = np.concatenate([U_synthetic, V_synthetic, C_synthetic, S_synthetic], axis=0)
print ("non_srvf_synthetics.shape: ", non_srvf_synthetics.shape)
non_srvf_pca_recovers = np.concatenate([U_recovered, V_recovered, C_recovered, S_recovered], axis=0)
print ("non_srvf_recovers.shape: ", non_srvf_pca_recovers.shape)

C_synthetic_curvatures, C_synthetic_torsions = compute_synthetic_curvature_and_torsion(C_recovered)
U_synthetic_curvatures, U_synthetic_torsions = compute_synthetic_curvature_and_torsion(U_recovered)
V_synthetic_curvatures, V_synthetic_torsions = compute_synthetic_curvature_and_torsion(V_recovered)
S_synthetic_curvatures, S_synthetic_torsions = compute_synthetic_curvature_and_torsion(S_recovered)

############
# 绘制group内的曲率和扭率对比全体的偏离程度的散点图
fig = plt.figure(dpi=300, figsize=(9,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(U_synthetic_curvatures,axis=0), color="r",marker="o", alpha=0.5, label='U')
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(V_synthetic_curvatures,axis=0), color="g",marker="s", alpha=0.5, label='V')
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(C_synthetic_curvatures,axis=0), color="b",marker="^", alpha=0.5, label='C')
ax1.scatter(np.mean(Curvatures,axis=0),np.mean(S_synthetic_curvatures,axis=0), color="orange",marker="*", alpha=0.5, label='S')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(U_synthetic_torsions,axis=0), color="r",marker="o", alpha=0.5, label='U')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(V_synthetic_torsions,axis=0), color="g",marker="s", alpha=0.5, label='V')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(C_synthetic_torsions,axis=0), color="b",marker="^", alpha=0.5, label='C')
ax2.scatter(np.mean(Torsions,axis=0),np.mean(S_synthetic_torsions,axis=0), color="orange",marker="*", alpha=0.5, label='S')
# 获取ax1的x轴和y轴范围
x1_min, x1_max = ax1.get_xlim()
y1_min, y1_max = ax1.get_ylim()
# 确保对角线从左下角连接到右上角
diag_min_1 = max(x1_min, y1_min)
diag_max_1 = min(x1_max, y1_max)
ax1.plot([diag_min_1, diag_max_1], [diag_min_1, diag_max_1], linestyle=":", color="k")
# 获取ax2的x轴和y轴范围
x2_min, x2_max = ax2.get_xlim()
y2_min, y2_max = ax2.get_ylim()
# 确保对角线从左下角连接到右上角
diag_min_2 = max(x2_min, y2_min)
diag_max_2 = min(x2_max, y2_max)
ax2.plot([diag_min_2, diag_max_2], [diag_min_2, diag_max_2], linestyle=":", color="k")
ax1.legend(loc='best') # 添加图例到子图ax1
ax2.legend(loc='best') # 添加图例到子图ax2
for ax in [ax1, ax2]:
    ax.grid(linestyle=":", alpha=0.5)
plt.savefig(geometry_dir + "/group_param_compare_Synthetic.png")
plt.close()

# 绘制合成曲线的curvature和torsion
##############################

####################为SRVF PCA绘制violinplot####################
# 创建一个DataFrame
df = pd.DataFrame(all_srvf_pca.train_res, columns=[f'PC{i+1}' for i in range(PCA_N_COMPONENTS)])
df['Type'] = Typevalues
# 创建一个4x4的子图网格
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
# 为每个主成分绘制violinplot
for i in range(PCA_N_COMPONENTS):
    ax = axes[i // 4, i % 4]
    sns.violinplot(x='Type', y=f'PC{i+1}', data=df, ax=ax, inner='quartile')  # inner='quartile' 在violin内部显示四分位数
    ax.set_title(f'Violinplot for Principal Component {i+1}')
    ax.set_ylabel('')  # 移除y轴标签，使得图更加简洁
plt.tight_layout()
plt.savefig(pca_anlysis_dir+"srvfPCA_total_Violinplot.png")
plt.close()

####################为srvf PCA绘制QQplot####################
unique_types = df['Type'].unique()
# 设置一个标记列表来区分每种类型
markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'v', '<', '>']  # ... you can extend this list if necessary
# 保证类型的数量不超过我们为其定义的标记数量
assert len(unique_types) <= len(markers), "Not enough markers defined!"
colors = sns.color_palette(n_colors=len(unique_types))
# 对于每个PC，进行QQ图分析
for pc in range(1, PCA_N_COMPONENTS+1):  # PC1 to PCn
    q = np.linspace(0, 1, 101)[1:-1]  # 从1%到99%
    quantiles_total = np.percentile(df[f'PC{pc}'], q*100)
    plt.figure(figsize=(8, 8))
    # 为每个Type绘制QQ图
    for idx, type_value in enumerate(unique_types):
        type_data = df[df['Type'] == type_value][f'PC{pc}']
        quantiles_type = np.percentile(type_data, q*100)
        # plt.scatter(quantiles_total, quantiles_type, label=type_value, color='black', marker=markers[idx])
        plt.scatter(quantiles_total, quantiles_type, 
                    label=type_value, 
                    color=colors[idx],
                    # marker="${}$".format(type_value),
                    marker = "x",
                    s=40)
    # 添加对角线
    plt.plot([min(quantiles_total), max(quantiles_total)], [min(quantiles_total), max(quantiles_total)], linestyle='--',color="dimgray", label='y=x line')
    plt.xlabel(f'Quantiles of Total Data (PC{pc})')
    plt.ylabel(f'Quantiles of Type Data (PC{pc})')
    plt.title(f'QQ plot: Types vs. Total for PC{pc}')
    plt.legend()
    plt.grid(True)
    plt.savefig(pca_anlysis_dir+f'QQ_plot_for_srvf_PC{pc}.png')
    plt.close()

# synthetics_from_kde = all_srvf_pca.train_kde.resample(5)
# synthetics = all_srvf_pca.pca.inverse_transform(synthetics_from_kde.T)*all_srvf_pca.train_std+all_srvf_pca.train_mean
# synthetics = synthetics.reshape(5,64,3)
# synthetic_curve = inverse_srvf(synthetics, np.zeros(3))
# plt.scatter(all_srvf_pca.train_res[:, 0], all_srvf_pca.train_res[:, 1], color="k")
# plt.scatter(synthetics_from_kde[:, 0], synthetics_from_kde[:, 1], color="r")
# for i in range(len(synthetic_curve)):
#     makeVtkFile(bkup_dir+"synthetic_curve{}.vtk".format(i), synthetic_curve[i], [],[])
#     plt.annotate("synthetic_curve{}".format(i), (synthetics_from_kde[i, 0], synthetics_from_kde[i, 1]))
# plt.savefig(pca_anlysis_dir+"synthetic_curve.png")

####################为坐标PCA绘制violinplot####################
# 创建一个DataFrame
df = pd.DataFrame(all_pca.train_res, columns=[f'PC{i+1}' for i in range(PCA_N_COMPONENTS)])
df['Type'] = Typevalues
# 创建一个4x4的子图网格
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
# 为每个主成分绘制violinplot
for i in range(PCA_N_COMPONENTS):
    ax = axes[i // 4, i % 4]
    sns.violinplot(x='Type', y=f'PC{i+1}', data=df, ax=ax, inner='quartile')  # inner='quartile' 在violin内部显示四分位数
    ax.set_title(f'Violinplot for Principal Component {i+1}')
    ax.set_ylabel('')  # 移除y轴标签，使得图更加简洁
plt.tight_layout()
plt.savefig(pca_anlysis_dir+"PCA_total_Violinplot.png")
plt.close()
####################为坐标PCA绘制QQplot####################
unique_types = df['Type'].unique()
# 设置一个标记列表来区分每种类型
markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'v', '<', '>']  # ... you can extend this list if necessary
# 保证类型的数量不超过我们为其定义的标记数量
assert len(unique_types) <= len(markers), "Not enough markers defined!"
colors = sns.color_palette(n_colors=len(unique_types))
# 对于每个PC，进行QQ图分析
for pc in range(1, PCA_N_COMPONENTS+1):  # PC1 to PCn
    q = np.linspace(0, 1, 101)[1:-1]  # 从1%到99%
    quantiles_total = np.percentile(df[f'PC{pc}'], q*100)
    plt.figure(figsize=(8, 8))
    # 为每个Type绘制QQ图
    for idx, type_value in enumerate(unique_types):
        type_data = df[df['Type'] == type_value][f'PC{pc}']
        quantiles_type = np.percentile(type_data, q*100)
        # plt.scatter(quantiles_total, quantiles_type, label=type_value, color='black', marker=markers[idx])
        plt.scatter(quantiles_total, quantiles_type, 
                    label=type_value, 
                    color=colors[idx],
                    # marker="${}$".format(type_value),
                    marker = "x",
                    s=40)
    # 添加对角线
    plt.plot([min(quantiles_total), max(quantiles_total)], [min(quantiles_total), max(quantiles_total)], linestyle='--',color="dimgray", label='y=x line')
    plt.xlabel(f'Quantiles of Total Data (PC{pc})')
    plt.ylabel(f'Quantiles of Type Data (PC{pc})')
    plt.title(f'QQ plot: Types vs. Total for PC{pc}')
    plt.legend()
    plt.grid(True)
    plt.savefig(pca_anlysis_dir+f'QQ_plot_for_PC{pc}.png')
    plt.close()


srvf_x_PC = 0
srvf_y_PC = 1 # 3
x_PC = 0
y_PC = 1 # 2

def get_main_color(palette):
    return palette[0]
# 取主颜色
main_colors = {key: get_main_color(value) for key, value in synthetic_cluster_colors.items()}
# 使用这个dictionary将Typevalues转换为颜色列表
color_values = [main_colors[t] for t in Typevalues]

# 创建图例的 handles 和 labels
legend_elements = [Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=main_colors[t], markersize=10, 
                          label=t) for t in ['U', 'C', 'V', 'S']]

fig = plt.figure(dpi=300, figsize=(6, 5))
ax1 = fig.add_subplot(111)
sc1 = ax1.scatter(all_srvf_pca.train_res[:, srvf_x_PC], all_srvf_pca.train_res[:, srvf_y_PC],
                  c=color_values,
                  alpha=0.7,
                  edgecolors='white')
ax1.grid(linestyle='--', linewidth=0.5)
ax1.set_xlabel("PC{}".format(srvf_x_PC+1))
ax1.set_ylabel("PC{}".format(srvf_y_PC+1))
ax1.legend(handles=legend_elements)  # 添加图例
plt.savefig(pca_anlysis_dir+"srvf_PCA_total.png")
plt.close()

fig = plt.figure(dpi=300, figsize=(6, 5))
ax2 = fig.add_subplot(111)
sc2 = ax2.scatter(all_pca.train_res[:, x_PC], all_pca.train_res[:, y_PC],
                  c=color_values,
                  alpha=0.7,
                  edgecolors='white')

ax2.set_xlabel("PC{}".format(x_PC+1))
ax2.set_ylabel("PC{}".format(y_PC+1))
ax2.grid(linestyle='--', linewidth=0.5)
ax2.legend(handles=legend_elements)  # 添加图例
plt.savefig(pca_anlysis_dir+"PCA_total.png")
plt.close()

log.write("PCA standardization: {}\n".format(PCA_STANDARDIZATION))
print ("所有PCA的标准化状态：", PCA_STANDARDIZATION)
#

###########################################################
######### Kernel PCA.通过KernelPCA或其他核技术，自定义一个核，使数据在新的特征空间中更均匀地分布。
# gamma=0.005
# array_0 = np.zeros(1000)
# array_1 = np.ones(1000)
# array_2 = np.ones(1000) * 2
# array_3 = np.ones(1000) * 3

# final_array = np.concatenate([array_0, array_1, array_2, array_3])
# kpca = KernelPCA(n_components=3, kernel='rbf', gamma=gamma)
# K = rbf_kernel(srvf_synthetics, gamma=gamma)
# eigenvalues = np.linalg.eigvalsh(K)
# print("EIGENVALUES:", eigenvalues)
# X_kpca = kpca.fit_transform(srvf_synthetics)
# # 绘图
# plt.figure(figsize=(8,6))
# plt.scatter(X_kpca[:, 0], X_kpca[:, 2], c=final_array, alpha=1, cmap=plt.cm.get_cmap('rainbow', 4))
# plt.title('Kernel PCA (gamma:{})'.format(gamma))
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend()  # 添加图例
# plt.savefig(pca_anlysis_dir+"srvf_kernel_PCA_total.png")

# gamma=0.01
# kpca = KernelPCA(n_components=3, kernel='rbf', gamma=gamma)
# K = rbf_kernel(non_srvf_synthetics , gamma=gamma)
# eigenvalues = np.linalg.eigvalsh(K)
# print("EIGENVALUES:", eigenvalues)
# X_kpca = kpca.fit_transform(non_srvf_synthetics )
# # 绘图
# plt.figure(figsize=(8,6))
# plt.scatter(X_kpca[:, 0], X_kpca[:, 2], c=final_array, alpha=1, cmap=plt.cm.get_cmap('rainbow', 4))
# plt.title('Kernel PCA (gamma:{})'.format(gamma))
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend()  # 添加图例
# plt.savefig(pca_anlysis_dir+"kernel_PCA_total.png")
#########
###########################################################

print ("开始计算geodesic距离")
total_geod_dist = []
total_type_pair = []
total_pca_dist  = []
total_srvf_pca_dist = []
for i in range(len(Procrustes_curves)):
    for j in range(i+1, len(Procrustes_curves)):
        geodesic_d = compute_geodesic_dist_between_two_curves(Procrustes_curves[i], Procrustes_curves[j])
        pca_dist = np.linalg.norm(all_pca.train_res[i]-all_pca.train_res[j])
        srvf_pca_dist = np.linalg.norm(all_srvf_pca.train_res[i]-all_srvf_pca.train_res[j])
        total_geod_dist.append(geodesic_d)
        total_type_pair.append([Typevalues[i], Typevalues[j]])
        total_pca_dist.append(pca_dist)
        total_srvf_pca_dist.append(srvf_pca_dist)
total_geod_dist = np.array(total_geod_dist)
total_pca_dist = np.array(total_pca_dist)
total_srvf_pca_dist = np.array(total_srvf_pca_dist)

pca_correlation = np.corrcoef(total_pca_dist, total_geod_dist)[0,1] 
srvf_pca_correlation = np.corrcoef(total_srvf_pca_dist, total_geod_dist)[0,1]
log.write("PCA correlation matrix: "+ str(pca_correlation)+"\n")
log.write("SRVF PCA correlation matrix: "+str(srvf_pca_correlation)+"\n")

srvf_synthetic_geod_dist = []
srvf_pca_dist = []
sampling_range = np.concatenate([range(0,100), range(1000,1100), range(2000,2100), range(3000,3100)])

for i in sampling_range:
    for j in sampling_range:
        geodesic_d = compute_geodesic_dist_between_two_curves(srvf_recovers[i], srvf_recovers[j])
        pca_dist = np.linalg.norm(srvf_synthetics[i]-srvf_synthetics[j])
        srvf_synthetic_geod_dist.append(geodesic_d)
        srvf_pca_dist.append(pca_dist)
srvf_synthetic_geod_dist = np.array(srvf_synthetic_geod_dist)
srvf_pca_dist = np.array(srvf_pca_dist)
# print ("srvf synthetic geodesic distance shape: ", srvf_synthetic_geod_dist.shape)
# print ("srvf pca distance shape: ", srvf_pca_dist.shape)
srvf_pca_correlation = np.corrcoef(srvf_pca_dist, srvf_synthetic_geod_dist)[0,1]
log.write("SRVF PCA correlation matrix (synthetic) : "+str(srvf_pca_correlation)+"\n")

non_srvf_geod_dist = []
non_srvf_pca_dist = []
for i in sampling_range:
    for j in sampling_range:
        geodesic_d = compute_geodesic_dist_between_two_curves(non_srvf_pca_recovers[i], non_srvf_pca_recovers[j])
        pca_dist = np.linalg.norm(non_srvf_synthetics[i]-non_srvf_synthetics[j])
        non_srvf_geod_dist.append(geodesic_d)
        non_srvf_pca_dist.append(pca_dist)
non_srvf_geod_dist = np.array(non_srvf_geod_dist)
non_srvf_pca_dist = np.array(non_srvf_pca_dist)
# print ("non_srvf geodesic distance shape: ", non_srvf_geod_dist.shape)
# print ("non_srvf pca distance shape: ", non_srvf_pca_dist.shape)
non_srvf_pca_correlation = np.corrcoef(non_srvf_pca_dist, non_srvf_geod_dist)[0,1]
log.write("Non-SRVF PCA correlation matrix (synthetic) : "+str(non_srvf_pca_correlation)+"\n")



"""
for loop in range(1):
    procrustes_curves = np.copy(Procrustes_curves)
    procs_srvf_curves = np.copy(Procs_srvf_curves)
    files = np.copy(Files)
    curvatures = np.copy(Curvatures)
    torsions = np.copy(Torsions)
    # 创建一个随机排列的索引
    indices = np.random.permutation(len(files))
    # 使用这个索引来重新排列 srvf_curves 和 files

    procrustes_curves = np.take(procrustes_curves, indices, axis=0)
    procs_srvf_curves = np.take(procs_srvf_curves, indices, axis=0)
    files = np.take(files, indices, axis=0)
    curvatures = np.take(curvatures, indices, axis=0)
    torsions = np.take(torsions, indices, axis=0)
    # save_shuffled_path = mkdir(bkup_dir,"shuffled_srvf_curves")
    save_shuffled_path = np.copy(bkup_dir)
    
    Scores = []

    # 获取当前时间
    now = datetime.now()

    # 将时间格式化为 'yymmddhhmmss' 格式
    formatted_time = now.strftime('%y%m%d%H%M%S')
    save_new_shuffle = mkdir(bkup_dir,formatted_time)
    loop_log = open(save_new_shuffle+"log.md", "w")

    np.save(save_new_shuffle + "file_indice.npy",files, allow_pickle=True)
    np.save(save_new_shuffle + "procrustes_curves.npy",procrustes_curves, allow_pickle=True)
    np.save(save_new_shuffle + "procs_srvf_curves.npy",procs_srvf_curves, allow_pickle=True)
    np.save(save_new_shuffle + "curvatures.npy",curvatures, allow_pickle=True)
    np.save(save_new_shuffle + "torsions.npy",torsions, allow_pickle=True)
    inverse_data_dir = mkdir(save_new_shuffle, "inverse_data")
    train_num = int(len(files)*0.75)
    test_num = int(len(files)-train_num)
    loop_log.write("# Train and test dataset split\n")
    loop_log.write("- train_num: {}\n".format(train_num))
    loop_log.write("- test_num: {}\n".format(test_num))
    train_procrustes_geodesic_d = compute_geodesic_dist(procrustes_curves[:train_num])
    test_procrustes_geodesic_d = compute_geodesic_dist(procrustes_curves[train_num:], True, np.mean(procrustes_curves[:train_num], axis=0))
    train_srvf_procrustes_geo_d = compute_geodesic_dist(procs_srvf_curves[:train_num])
    test_srvf_procrustes_geo_d = compute_geodesic_dist(procs_srvf_curves[train_num:],True, np.mean(procs_srvf_curves[:train_num], axis=0))
    train_files = files[:train_num]
    test_files = files[train_num:]

    # To-Do:Standardization
    data_dict = {
        'Procrustes_aligned': [procrustes_curves[:train_num], procrustes_curves[train_num:]],
        'Procrustes_aligned_SRVF': [procs_srvf_curves[:train_num], procs_srvf_curves[train_num:]]
    }
    param_dict = {
        'Curvature': [curvatures[:train_num], curvatures[train_num:]],
        'Torsion': [torsions[:train_num], torsions[train_num:]],
    }
    dist_dict = {
    'Procrustes_geodesic_dist': [train_procrustes_geodesic_d, test_procrustes_geodesic_d],
    'SRVF_Procrustes_geodesic_dist': [train_srvf_procrustes_geo_d, test_srvf_procrustes_geo_d]
    }
    

    loop_log.write("***\n")
    ###############################################
    # PCA
    loop_log.write("np.corrcoef is Pearson Correlation Coefficient which measures linear correlation.\n")
    loop_log.write("# coord PCA\n")
    coord_PCAs = []
    for data_key, data_values in data_dict.items():
        loop_log.write("## "+data_key+"\n")
        loop_log.write("- PCA_training_and_test will standardize data automatically.")
        train_data, test_data = data_values  # 取出列表中的两个值
        train_data=train_data.reshape(train_num,-1)
        test_data=test_data.reshape(test_num,-1)
        coord_PCAs.append(PCAHandler(train_data, test_data,standardization=PCA_STANDARDIZATION))
        coord_PCAs[-1].PCA_training_and_test()
        components_figname = save_new_shuffle+"coord_componentse_{}.png".format(data_key)
        coord_PCAs[-1].visualize_results(components_figname)
        loading_figname = "coord_{}.png".format(data_key)
        loop_log.write("![coord_{}]({})\n".format(data_key,"./"+loading_figname))
        loop_log.write("![coord_{}]({})\n".format(data_key,"./"+components_figname))
        coord_PCAs[-1].visualize_loadings(dist_dict = dist_dict, save_path=save_new_shuffle+loading_figname)
        train_inverse = coord_PCAs[-1].inverse_transform_from_loadings(coord_PCAs[-1].train_res).reshape(train_num, -1, 3)
        test_inverse = coord_PCAs[-1].inverse_transform_from_loadings(coord_PCAs[-1].test_res).reshape(test_num, -1, 3)
        process_data_key(data_key, train_inverse, test_inverse, train_files, test_files, inverse_data_dir)
        coord_PCAs[-1].plot_scatter_kde(save_new_shuffle+"coord_scatter_kde_{}.png".format(data_key))
        coord_PCAs[-1].compute_kde()
        print ("Coords,", data_key, "JS divergence:", coord_PCAs[-1].compute_train_test_js_divergence())
        for dist_key, dist_values in dist_dict.items():
            Scores.append(ScoreHandler(data_name="Coords"+data_key, dist_name=dist_key, dist=dist_values[0], pca_result=coord_PCAs[-1].train_res, train=1))
            Scores.append(ScoreHandler(data_name="Coords"+data_key, dist_name=dist_key, dist=dist_values[1], pca_result=coord_PCAs[-1].test_res, train=0))
            
    loop_log.write("***\n")

    ###############################################
    # separate x, y, z coordinates
    loop_log.write("***\n")
    loop_log.write("# United PCA\n")
    # xyz =["x", "y", "z"]
    united_PCAs = []
    for data_key, data_values in data_dict.items():
        loop_log.write("## "+data_key+"\n")
        train_data, test_data = data_values  # 取出列表中的两个值
        train_res = []
        test_res = []
        united_internal_PCAs = []
        for i in range(3):
            united_internal_PCAs.append(PCAHandler(train_data[:,:,i], test_data[:,:,i],standardization=PCA_STANDARDIZATION))
            united_internal_PCAs[-1].PCA_training_and_test()
            train_res_temp, test_res_temp = united_internal_PCAs[-1].train_res, united_internal_PCAs[-1].test_res
            train_res.append(train_res_temp)
            test_res.append(test_res_temp)
        train_data = np.array(train_res).transpose(1,0,2).reshape(train_num, -1)
        test_data = np.array(test_res).transpose(1,0,2).reshape(test_num, -1)
        united_PCAs.append(PCAHandler(train_data, test_data, standardization=PCA_STANDARDIZATION))
        united_PCAs[-1].PCA_training_and_test()
        components_figname = save_new_shuffle+"united_componentse_{}.png".format(data_key)
        united_PCAs[-1].visualize_results(components_figname)
        loading_figname = "united_{}.png".format(data_key)
        loop_log.write("![united_{}]({})\n".format(data_key,"./"+loading_figname))
        loop_log.write("![united_{}]({})\n".format(data_key,"./"+components_figname))
        united_PCAs[-1].visualize_loadings(dist_dict = dist_dict, save_path=save_new_shuffle+loading_figname)
        train_inverse_large = united_PCAs[-1].inverse_transform_from_loadings(united_PCAs[-1].train_res)
        test_inverse_large = united_PCAs[-1].inverse_transform_from_loadings(united_PCAs[-1].test_res)
        train_inverse_parts = np.split(train_inverse_large, 3, axis=1)  # 分解为三个部分
        test_inverse_parts = np.split(test_inverse_large, 3, axis=1)  # 分解为三个部分
        train_inverse = []
        test_inverse = []
        for i in range(3):
            train_inverse_temp = united_internal_PCAs[i].inverse_transform_from_loadings(train_inverse_parts[i])
            test_inverse_temp = united_internal_PCAs[i].inverse_transform_from_loadings(test_inverse_parts[i])
            train_inverse.append(train_inverse_temp)
            test_inverse.append(test_inverse_temp)
        train_data_inverse = np.array(train_inverse).transpose(1,2,0)
        test_data_inverse = np.array(test_inverse).transpose(1,2,0)
        train_data_inverse = recovered_curves(train_data_inverse,"SRVF" in data_key)
        inverse_dir = mkdir(inverse_data_dir, "united_"+data_key)
        write_curves_to_vtk(train_data_inverse, train_files, inverse_dir+"train_inverse_{}.vtk".format(data_key))
        write_curves_to_vtk(test_data_inverse, test_files, inverse_dir+"test_inverse_{}.vtk".format(data_key))
        united_PCAs[-1].plot_scatter_kde(save_new_shuffle+"united_scatter_kde_{}.png".format(data_key))
        united_PCAs[-1].compute_kde()
        print ("United,",data_key, "JS divergence:", united_PCAs[-1].compute_train_test_js_divergence())
        for dist_key, dist_values in dist_dict.items():
            Scores.append(ScoreHandler(data_name="United_"+data_key, dist_name=dist_key, dist=dist_values[0], pca_result=united_PCAs[-1].train_res, train=1))
            Scores.append(ScoreHandler(data_name="United_"+data_key, dist_name=dist_key, dist=dist_values[1], pca_result=united_PCAs[-1].test_res, train=0))

    ###############################################
    loop_log.write("# Geometric param PCA\n")
    param_PCAs = []
    for data_key, data_values in param_dict.items():
        loop_log.write("## "+data_key+"\n")
        train_data, test_data = data_values  # 取出列表中的两个值
        train_data = train_data + np.random.normal(0, smooth_scale, train_data.shape)
        test_data = test_data + np.random.normal(0, smooth_scale, test_data.shape)
        loop_log.write("- PCA_training_and_test will standardize data automatically.\n")
        loop_log.write("- PCA_training_and_test will add a small amount of noise to the data.\n")
        param_PCAs.append(PCAHandler(train_data, test_data,standardization=PCA_STANDARDIZATION))
        param_PCAs[-1].PCA_training_and_test()
        components_figname = save_new_shuffle+"param_componentse_{}.png".format(data_key)
        param_PCAs[-1].visualize_results(components_figname)
        loading_figname = "param_{}.png".format(data_key)
        loop_log.write("![param_{}]({})\n".format(data_key,"./"+loading_figname))
        loop_log.write("![param_{}]({})\n".format(data_key,"./"+components_figname))
        param_PCAs[-1].visualize_loadings(dist_dict = dist_dict, save_path=save_new_shuffle+loading_figname)
        param_PCAs[-1].plot_scatter_kde(save_new_shuffle+"param_scatter_kde_{}.png".format(data_key))
        param_PCAs[-1].compute_kde()
        print ("Param,",data_key, "JS divergence:", param_PCAs[-1].compute_train_test_js_divergence())
        for dist_key, dist_values in dist_dict.items():
            Scores.append(ScoreHandler(data_name="Param_"+data_key, dist_name=dist_key, dist=dist_values[0], pca_result=param_PCAs[-1].train_res, train=1))
            Scores.append(ScoreHandler(data_name="Param_"+data_key, dist_name=dist_key, dist=dist_values[1], pca_result=param_PCAs[-1].test_res, train=0))
    loop_log.write("***\n")

    ###############################################
    loop_log.write("***\n")
    loop_log.close()

    score_file = open(save_new_shuffle+"scores.csv", "w")
    for i in range(0, len(Scores), 2): # 2是为了区别test和train
        score_file.write("Train,")
        for score_name, score_value in Scores[i].score.items():
            score_file.write("{},".format(score_value))
        score_file.write("\n")
        score_file.write("Test,")
        for score_name, score_value in Scores[i+1].score.items():
            score_file.write("{},".format(score_value))
        score_file.write("\n")

    score_file.close()


score_files = glob.glob(bkup_dir+"shuffled_srvf_curves/*/scores.csv")
# 定义一个列表来存储每个文件的内容
all_lines = []
# 遍历score_files中的每个文件路径
for file_path in score_files:
    # 打开文件，读取每一行的内容并将其存储在列表中
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        all_lines.append(lines)

# 检查所有文件的行数是否相等
if len(set(len(lines) for lines in all_lines)) > 1:
    print('所有文件的行数不等，不能合并。')
    exit()

# 创建一个新的文件来存储合并后的内容
with open(bkup_dir+'merged_file.csv', 'w', encoding='utf-8') as mf:
    # 获取all_lines的转置，这样我们可以每次迭代一行，而不是一个文件
    for lines in zip(*all_lines):
        # 删除每一行末尾的换行符，然后将所有行内容合并，并在其后添加一个换行符
        merged_line = ''.join(line.rstrip('\n') for line in lines) + '\n'
        # 将合并后的行写入新的文件中
        mf.write(merged_line)
"""
end_time = datetime.now()
total_time = end_time - start_time
print(dir_formatted_time, "is done in", total_time.seconds, "seconds.")
log.close()
open_folder_in_explorer(bkup_dir)