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


warnings.filterwarnings("ignore")

PCA_N_COMPONENTS = 16
Multi_plot_rows = 4
SCALETO1 = False
PCA_STANDARDIZATION = 1
ORIGINAL_GEO_PARAM = False
USE_REAL_DATA_FOR_GEO_PARAM = False


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
pre_Curvatures = []
pre_Torsions = []
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
    pre_Curvatures.append(sma_curv[::-1])
    sma_tors = np.convolve(Tors, weights, 'valid')
    pre_Torsions.append(sma_tors[::-1])
unaligned_curves = np.array(unaligned_curves)
geometry_dir = mkdir(bkup_dir, "geometry")
Typevalues = np.array(Typevalues)

if SCALETO1:
    for i in range(len(unaligned_curves)):
        unaligned_curves[i] = unaligned_curves[i]*(1.0/measure_length(unaligned_curves[i]))

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
i=30 # U
j=46 # S


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

# SRVF计算
Procs_srvf_curves = np.zeros_like(Procrustes_curves)
for i in range(len(Procrustes_curves)):
    Procs_srvf_curves[i] = calculate_srvf(Procrustes_curves[i])

makeVtkFile(bkup_dir+"mean_curve.vtk", np.mean(Procrustes_curves,axis=0),[],[] )
mean_srvf_inverse = inverse_srvf(np.mean(Procs_srvf_curves,axis=0),np.zeros(3))
makeVtkFile(bkup_dir+"mean_srvf.vtk", mean_srvf_inverse,[],[] )
contains_nan = np.isnan(Procs_srvf_curves).any()

print(f"Procs_srvf_curves contains NaN values: {contains_nan}")
#####
# 绘制一个4子图的plot，有对齐后的4个类别的曲线和SRVF曲线标注
# plot_curves_with_arrows(1, 2, Procrustes_curves, 
#                         Procs_srvf_curves, 
#                         Typevalues, 
#                         geometry_dir + "/Procrustes_curves_with_srvf.png")
#####

#################################
# frechet_mean_srvf = compute_frechet_mean(Procs_srvf_curves)
# frechet_mean_srvf = frechet_mean_srvf / measure_length(frechet_mean_srvf)
# 计算PCA
# 保存数据
all_srvf_pca = PCAHandler(Procs_srvf_curves.reshape(len(Procs_srvf_curves),-1), None, PCA_N_COMPONENTS, PCA_STANDARDIZATION)
all_srvf_pca.PCA_training_and_test()
all_srvf_pca.compute_kde()
joblib.dump(all_srvf_pca.pca, bkup_dir + 'srvf_pca_model.pkl')
np.save(bkup_dir+"pca_model_filename.npy",Files )
print ("saved pca model to", bkup_dir + 'srvf_pca_model.pkl')
log.write("CCR:"+str(np.sum(all_srvf_pca.pca.explained_variance_ratio_))+"\n")
pca_anlysis_dir = mkdir(bkup_dir, "pca_analysis")
pca_components_figname = pca_anlysis_dir+"pca_plot_variance.png"
# all_pca.visualize_results(pca_components_figname)
srvf_pca_components_figname = pca_anlysis_dir + "srvf_pca_plot_variance.png"
all_srvf_pca.visualize_results(srvf_pca_components_figname)
print ("debug1")


#################################
# 在合理范围内，每个mode变化时，曲线上的哪些landmark的欧几里得距离变化最大
# 同样，哪些landmark的curvature和torsion变化最大
# 显然landmark欧几里得距离的变化应当是线性的，但curvature torsion则未必

# FrechetMean = compute_frechet_mean(Procrustes_curves)
# print ("FrechetMean.shape:", FrechetMean.shape)

flatten_curvatures = []
flatten_torsions = []
for _ in [0]:
    FrechetMean = compute_frechet_mean(Procrustes_curves)
    FrechetMean_srvf = calculate_srvf(FrechetMean)
    # print ("FrechetMean_srvf.shape:", FrechetMean_srvf.shape)
    flatten_FrechetMean_srvf = FrechetMean_srvf.reshape(-1, ).reshape(1, 192)
    flatten_FrechetMean_srvf_normalized = (flatten_FrechetMean_srvf - all_srvf_pca.train_mean) / all_srvf_pca.train_std
    frechet_feature = all_srvf_pca.pca.transform(flatten_FrechetMean_srvf_normalized)
    print ("frechet_feature:", frechet_feature)
    # 日后这个初始位置还要变化，看在不同点上变化规律是否一致

    fig1, axes1 = plt.subplots((PCA_N_COMPONENTS // Multi_plot_rows), Multi_plot_rows, figsize=(15, 15))
    fig2, axes2 = plt.subplots((PCA_N_COMPONENTS // Multi_plot_rows), Multi_plot_rows, figsize=(15, 15))
    fig3, axes3 = plt.subplots((PCA_N_COMPONENTS // Multi_plot_rows), Multi_plot_rows, figsize=(15, 15))
    fig4, axes4 = plt.subplots((PCA_N_COMPONENTS // Multi_plot_rows), Multi_plot_rows, figsize=(15, 15))

    # 用于存储所有形状的列表
    all_delta_shapes = []

    for pc in range(PCA_N_COMPONENTS):
        i = pc // Multi_plot_rows
        j = pc % Multi_plot_rows
        ax1 = axes1[i][j]
        ax2 = axes2[i][j]
        ax3 = axes3[i][j]
        ax4 = axes4[i][j]
        delta_range = np.linspace(-np.std(all_srvf_pca.train_res[:, pc]), np.std(all_srvf_pca.train_res[:, pc]), 11)

        # 用于临时存储当前 PC 的所有形状
        temp_delta_shapes = []

        for delta in delta_range:
            new_frechet_feature = copy.deepcopy(frechet_feature)
            new_frechet_feature[0][pc] += delta
            new_frechet_srvf = all_srvf_pca.inverse_transform_from_loadings(new_frechet_feature).reshape(1, -1, 3)
            new_frechet = recovered_curves(new_frechet_srvf, True)[0]
            temp_delta_shapes.append(new_frechet)

        # 对齐所有形状
        a_curves = align_icp(temp_delta_shapes, base_id=0)
        Procrustes_delta_curves = align_procrustes(a_curves, base_id=0)
        parametrized_curves = np.zeros_like(Procrustes_delta_curves)
        for i in range(len(Procrustes_delta_curves)):
            parametrized_curves[i] = arc_length_parametrize(Procrustes_delta_curves[i])
        aligned_shapes = np.array(parametrized_curves)

        # 计算距离，曲率和扭率
        total_delta_dist = []
        total_delta_curvatures = []
        total_delta_torsions = []

        prev_curvature = None
        prev_torsion = None

        total_curvature_energy = []
        total_torsion_energy = []
        for idx, shape in enumerate(aligned_shapes):
            d_curvature, d_torsion = compute_curvature_and_torsion(shape)
            d_energy_curvature, d_energy_torsion = compute_geometry_param_energy(d_curvature, d_torsion)
            total_curvature_energy.append(d_energy_curvature)
            total_torsion_energy.append(d_energy_torsion)
            delta_dist = np.zeros(len(shape))

            if idx > 0:
                for lm in range(len(shape)):
                    delta_dist[lm] = np.linalg.norm(shape[lm] - aligned_shapes[idx - 1][lm])
                total_delta_dist.append(delta_dist)

                # 计算曲率和扭率的差
                delta_curvature = d_curvature - prev_curvature
                delta_torsion = d_torsion - prev_torsion
                total_delta_curvatures.append(delta_curvature)
                total_delta_torsions.append(delta_torsion)

            prev_curvature = d_curvature
            prev_torsion = d_torsion

        # 将形状添加到总列表
        all_delta_shapes.extend(aligned_shapes)

        total_delta_dist = np.array(total_delta_dist)
        total_delta_curvatures = np.array(total_delta_curvatures)
        total_delta_torsions = np.array(total_delta_torsions)

        sns.heatmap(total_delta_dist, cmap="mako", ax=ax1, cbar=True)
        sns.heatmap(total_delta_curvatures, cmap="mako", ax=ax2, cbar=True)
        sns.heatmap(total_delta_torsions, cmap="mako", ax=ax3, cbar=True)
        ax4.plot(total_curvature_energy, label="curvature", linestyle="-",color="k")
        ax4.plot(total_torsion_energy, label="torsion", linestyle="--",color="k")

        ax1.set_title(f"PC{pc}")
        ax2.set_title(f"PC{pc}")
        ax3.set_title(f"PC{pc}")
        flatten_curvatures.append(total_delta_curvatures.flatten())
        flatten_torsions.append(total_delta_torsions.flatten())

    fig1.savefig(geometry_dir + "delta_dist(coordinates)_pc.png")
    fig2.savefig(geometry_dir + "delta_dist(curvature)_pc.png")
    fig3.savefig(geometry_dir + "delta_dist(torsion)_pc.png")
    fig4.savefig(geometry_dir + "delta_dist(energy)_pc.png")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

fig1 = plt.figure(figsize=(15, 15))
fig2 = plt.figure(figsize=(15, 15))
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
for i in range(len(flatten_curvatures)):
    ax1.plot(flatten_curvatures[i], label=f"C{i}",linewidth=0.5)
    ax2.plot(flatten_torsions[i], label=f"T{i}",linewidth=0.5)
    # plt.plot(flatten_torsions[i], label=f"T{i}")
fig1.savefig(geometry_dir + "delta_dist(curvature)_all.png")
fig2.savefig(geometry_dir + "delta_dist(torsion)_all.png")





















###############

log.write("RECONSTRUCT_WITH_SRVF:"+str(1)+"\n")
OG_data_inverse = all_srvf_pca.inverse_transform_from_loadings(all_srvf_pca.train_res).reshape(len(all_srvf_pca.train_res), -1, 3)
OG_data_inverse = recovered_curves(OG_data_inverse, 1)
geo_dist_OG_to_reverse = []
length_reverse = []
for i in range(len(OG_data_inverse)):
    geo_dist_OG_to_reverse.append(compute_geodesic_dist_between_two_curves(Procrustes_curves[i], OG_data_inverse[i]))
    length_reverse.append(measure_length(OG_data_inverse[i]))
log.write("MEAN geo_dist_OG_to_reverse:"+str(np.mean(geo_dist_OG_to_reverse))+"\n")
log.write("STD geo_dist_OG_to_reverse:"+str(np.std(geo_dist_OG_to_reverse))+"\n")
log.write("MEAN length_reverse:"+str(np.mean(length_reverse))+"\n")
log.write("STD length_reverse:"+str(np.std(length_reverse))+"\n")
if ORIGINAL_GEO_PARAM:
    Curvatures, Torsions = compute_synthetic_curvature_and_torsion(Procrustes_curves,weights)
else:
    Curvatures, Torsions = compute_synthetic_curvature_and_torsion(OG_data_inverse,weights)

# average_of_means_torsions = np.mean([np.mean(tors) for tors in Torsions])
average_of_means_torsions = np.mean([np.mean(np.abs(tors)) for tors in Torsions])
average_of_std_torsions = np.mean([np.std(tors) for tors in Torsions])

torsion_param_group = []
param_group = []
HT_group = []
LT_group = []
for i in range(len(Torsions)):
    if np.mean(np.abs(Torsions[i])) > average_of_means_torsions and np.std(Torsions[i]) > average_of_std_torsions:
        torsion_param_group.append("HMHS")
        param_group.append("HT")
        HT_group.append(i)
    elif np.mean(np.abs(Torsions[i])) > average_of_means_torsions and np.std(Torsions[i]) <= average_of_std_torsions:
        torsion_param_group.append("HMLS")
        param_group.append("HT")
        HT_group.append(i)
    elif np.mean(np.abs(Torsions[i])) <= average_of_means_torsions and np.std(Torsions[i]) > average_of_std_torsions:
        torsion_param_group.append("LMHS")
        param_group.append("LT")
        LT_group.append(i)
    elif np.mean(np.abs(Torsions[i])) <= average_of_means_torsions and np.std(Torsions[i]) <= average_of_std_torsions:
        torsion_param_group.append("LMLS")
        param_group.append("LT")
        LT_group.append(i)

# 输出结果
for label, count in Counter(torsion_param_group).items():
    print(f"{label}: {count}")
print ("average_of_means_torsions:", average_of_means_torsions)

LT_curvatures = Curvatures[LT_group]
HT_curvatures = Curvatures[HT_group]
print ("len(LT_curvatures):", len(LT_curvatures))
print ("len(HT_curvatures):", len(HT_curvatures))
average_of_LT_curvature = np.mean([np.mean(curv) for curv in LT_curvatures])
average_of_HT_curvature = np.mean([np.mean(curv) for curv in HT_curvatures])
print ("average_of_LT_curvature:", average_of_LT_curvature)
print ("average_of_HT_curvature:", average_of_HT_curvature)


quad_param_group = []
for i in range(len(Curvatures)):
    curvature_mean = np.mean(Curvatures[i])
    if param_group[i] == 'LT':
        threshold = average_of_LT_curvature
    elif param_group[i] == 'HT':
        threshold = average_of_HT_curvature
    if curvature_mean > threshold:
        quad_param_group.append(param_group[i] + 'HC')
        param_group[i] = torsion_param_group[i] + 'HC'
    else:
        quad_param_group.append(param_group[i] + 'LC')
        param_group[i] = torsion_param_group[i] + 'LC'


import matplotlib.pyplot as plt
from collections import defaultdict

# 给定的代码
counter = defaultdict(lambda: defaultdict(int))
overall_counter = defaultdict(int)

# 遍历每个数据点并更新计数器
for type_val, param in zip(Typevalues, param_group):
    counter[type_val][param] += 1
    overall_counter[param] += 1

# 绘制柱状图
labels = list(counter.keys())
labels.append("Overall")
type_vals = sorted(list({tv for inner_dict in counter.values() for tv in inner_dict.keys()}))

# 数据存储的列表
param2cusv_data_list = []
# 每个 Typevalue 的柱子位置
for idx, tv in enumerate(type_vals):
    counts = [counter[label][tv] for label in labels[:-1]]
    counts.append(overall_counter[tv])  # 添加全体数据的计数
    param2cusv_data_list.append(counts)

# 将数据转化为 DataFrame
param2cusv_df = pd.DataFrame(param2cusv_data_list, columns=labels, index=type_vals)
# 将数据保存为CSV文件
param2cusv_df.to_csv(bkup_dir+"param2cusv.csv")

param_group_unique_labels = list(set(quad_param_group))
# 初始化字典
param_dict = {label: {} for label in param_group_unique_labels}

# 填充字典
for label in param_group_unique_labels:
    # 使用布尔索引来选择与当前标签对应的数据
    selected_data_torsion = [Torsions[i] for i, tag in enumerate(quad_param_group) if tag == label]
    selected_data_curvature = [Curvatures[i] for i, tag in enumerate(quad_param_group) if tag == label]
    
    # 将选择的数据转换为numpy array并保存到字典中
    param_dict[label]['Torsion'] = np.array(selected_data_torsion)
    param_dict[label]['Curvature'] = np.array(selected_data_curvature)



def fit_kde(data):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(data)
    return kde


# 用smote法制造一些合成数据用来训练分类器
Synthetic_data = []
Synthetic_X = []
Synthetic_y = []
for tag in param_dict.keys():
    indices = [idx for idx, label in enumerate(quad_param_group) if label == tag]
    group_feature = np.array([all_srvf_pca.train_res[idx] for idx in indices])
    # print ("group_feature.shape:", group_feature.shape)
    group_kde = fit_kde(group_feature)
    sample_num = 1000
    synthetic_feature = group_kde.sample(sample_num)
    synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(synthetic_feature).reshape(sample_num, -1, 3)
    synthetic_recovered = recovered_curves(synthetic_inverse, True)
    Synthetic_data.extend(synthetic_recovered)
    synthetic_curvatures,synthetic_torsions = compute_synthetic_curvature_and_torsion(synthetic_recovered, weights)
    # print ("synthetic_curvatures.shape:", synthetic_curvatures.shape)
    # print ("synthetic_torsions.shape:", synthetic_torsions.shape)
    for torsion, curvature in zip(synthetic_torsions, synthetic_curvatures):
        c_energy, t_energy = compute_geometry_param_energy(curvature,torsion)
        # print ("c_energy:", c_energy, "t_energy:", t_energy)
        Synthetic_X.append([c_energy, t_energy])
    Synthetic_y.extend([tag] * sample_num)
    
Synthetic_data = np.array(Synthetic_data)
Synthetic_X = np.array(Synthetic_X)
Synthetic_y = np.array(Synthetic_y)
print ("Synthetic_data.shape:", Synthetic_data.shape)
print ("Synthetic_X.shape:", Synthetic_X.shape)
print ("Synthetic_y.shape:", Synthetic_y.shape)



# 对于每个标签，为其下的所有数据计算能量
# 准备数据
X = []  # 用于存储所有的能量值
y = []  # 用于存储对应的标签
for label in param_group_unique_labels:
    torsions = param_dict[label]['Torsion']
    curvatures = param_dict[label]['Curvature']
    
    # 初始化能量值列表
    energies = []
    
    # 为每个数据计算能量
    for torsion, curvature in zip(torsions, curvatures):
        energy = compute_geometry_param_energy(curvature, torsion)
        energies.append(energy)
    
    # 将计算的能量值存储在字典中
    param_dict[label]['Energy'] = energies
    X.extend(energies)
    y.extend([label] * len(energies))

# 定义颜色映射
colors = {
    label: plt.cm.Set3((i)/(len(param_group_unique_labels))) for i, label in enumerate(param_group_unique_labels)
}


# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(Synthetic_X, Synthetic_y, test_size=0.3, random_state=12, stratify=Synthetic_y)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# 定义Random Forests分类器
rf_clf = RandomForestClassifier(n_estimators=10, random_state=12)  # n_estimators代表决策树的数量

# 训练分类器
rf_clf.fit(X_train_scaled, y_train)

# 预测
y_pred = rf_clf.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 获取预测概率(分数)
y_prob = rf_clf.predict_proba(X_scaled)


# Create a figure and axis layout
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
# Flatten the axes for easy iteration
axes = axes.ravel()
# Plot histograms for each column of y_prob
for idx, ax in enumerate(axes):
    ax.hist(y_prob[:, idx], bins=50, color='blue', alpha=0.7)
    ax.set_title(f'Histogram of y_prob column {idx + 1}')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 1)
# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(bkup_dir+"y_prob_histogram.png")
plt.close()

y_prob_max = np.max(y_prob, axis=1)
print ("y_prob_max.shape:", y_prob_max.shape)
# 创建一个图形和轴

# 创建一个图形和轴
fig, ax = plt.subplots(dpi=300)

# 初始化一个索引来跟踪y_prob_max中的当前位置
index = 0

# 绘制散点图
for label in param_group_unique_labels:
    energies = param_dict[label]['Energy']
    curvatures, torsions = zip(*energies)
    
    # 获取当前标签对应的大小值
    sizes_for_label = y_prob_max[index : index + len(energies)]
    
    ax.scatter(curvatures, torsions, 
               color=colors[label], 
               label=label, 
               alpha=0.6, 
               s=sizes_for_label*sizes_for_label*75)  
    
    # 更新索引
    index += len(energies)

# 显示图形
ax.set_xlabel('Curvature Energy')
ax.set_ylabel('Torsion Energy')
ax.set_title('Energy Scatter Plot by Label')
ax.legend()
ax.grid(linestyle='--', alpha=0.5)
plt.savefig(bkup_dir+"Energy_Scatter_Plot_by_Label.png")
plt.close()



slopes = []
print ("(PCA_N_COMPONENTS//4):",(PCA_N_COMPONENTS//4))
fig, axes = plt.subplots((PCA_N_COMPONENTS//Multi_plot_rows), Multi_plot_rows, dpi=300, figsize=(16, 13))
for i in range(PCA_N_COMPONENTS//Multi_plot_rows):
    for j in range(Multi_plot_rows):
        ax = axes[i][j]
        ax.tick_params(axis='both', which='major', labelsize=8)
        for tag in param_dict.keys():
            indices = [idx for idx, label in enumerate(quad_param_group) if label == tag]
            selected_data = np.array([all_srvf_pca.train_res[idx] for idx in indices])
            param_feature = np.array(param_dict[tag]["Energy"])[:,0]/np.array(param_dict[tag]["Energy"])[:,1]

            # std_selected_data = selected_data[:,Multi_plot_rows*i+j]/np.std(selected_data[:,Multi_plot_rows*i+j])
            std_selected_data = np.tanh(selected_data[:,Multi_plot_rows*i+j]/np.std(selected_data[:,Multi_plot_rows*i+j]))

            ax.scatter(std_selected_data,param_feature,
                           color=colors[tag],
                           alpha=0.6, 
                           s=25)
            model = np.polyfit(std_selected_data, param_feature, 1)
            slope = model[0]
            slopes.append(slope)
            # print (tag," PC", 4*i+j, " slope:", slope)
            if abs(slope) > 0.03:
                linestyle = '-'
                linewidth = 2
            else:
                linestyle = ':'
                linewidth = 1
            predicted = np.poly1d(model)
            predict_range = np.linspace(np.min(std_selected_data), np.max(std_selected_data), 10)
            ax.plot(predict_range, 
                        predicted(predict_range), 
                        color=colors[tag], 
                        linewidth=linewidth,
                        linestyle=linestyle)
        # if j == 0:
        #     ax.set_ylabel(f'Row {i+1}')
        # if i == 3:
        #     ax.set_xlabel(f'Feature {4*i+j+1}')

# plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3))
plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.savefig(pca_anlysis_dir+"energy_VS_PCs.png")
plt.close()

print ("energy平均斜率：",np.mean(np.abs(slopes)))
print ("energy斜率标准差：",np.std(slopes))
slopes = np.array(slopes).reshape(-1,4)
print ("slopes.shape:", slopes.shape)
for i , tag in enumerate(param_dict.keys()):
    print (tag, "的支配性PC是:", np.max(np.abs(slopes[:,i])), np.argmax(np.abs(slopes[:,i])), "其平均绝对斜率是:",np.mean(np.abs(slopes[:,i])))


slopes = []
# y_prob 和 all_srvf_pca.train_res 的相关性分析
fig, axes = plt.subplots((PCA_N_COMPONENTS//Multi_plot_rows), Multi_plot_rows, figsize=(15, 15))
# 遍历all_srvf_pca.train_res的每一个特征
for i in range(PCA_N_COMPONENTS//Multi_plot_rows):
    for j in range(Multi_plot_rows):
        ax = axes[i][j]
        ax.tick_params(axis='both', which='major', labelsize=8)
        for tag in param_dict.keys():
            indices = [idx for idx, label in enumerate(quad_param_group) if label == tag]
            selected_data = np.array([all_srvf_pca.train_res[idx] for idx in indices])
            # std_selected_data = selected_data[:,Multi_plot_rows*i+j]/np.std(selected_data[:,Multi_plot_rows*i+j])
            std_selected_data = np.tanh(selected_data[:,Multi_plot_rows*i+j]/np.std(selected_data[:,Multi_plot_rows*i+j]))
            # prob_feature= np.max(y_prob[indices, :], axis=1)
            # prob_feature = y_prob[indices, ] # [数据的tag,分数的tag]
            # 初始化一个和y_prob形状相同的数组
            difference = np.zeros_like(y_prob[indices, :])

            # 对于y_prob的每一行
            for m, row in enumerate(y_prob[indices, :]):
                max_val = np.max(row)  # 找到最大值
                difference[m] = max_val - row  # 从最大值中减去每个元素

            # 如果你只关心每个样本的最大差异，可以这样得到
            prob_feature = np.max(difference, axis=1)
            # print ("prob_feature.shape:", prob_feature.shape)
            ax.scatter(std_selected_data,prob_feature,
                           color=colors[tag],
                           alpha=0.6, 
                           s=25)
            model = np.polyfit(std_selected_data, prob_feature, 1)
            slope = model[0]
            slopes.append(slope)
            # print (tag," PC", 4*i+j, " slope:", slope)
            if abs(slope) > 0.02:
                linestyle = '-'
                linewidth = 2
            else:
                linestyle = ':'
                linewidth = 1
            predicted = np.poly1d(model)
            predict_range = np.linspace(np.min(std_selected_data), np.max(std_selected_data), 10)
            ax.plot(predict_range, 
                        predicted(predict_range), 
                        color=colors[tag], 
                        linewidth=linewidth,
                        linestyle=linestyle)


plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3))
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(pca_anlysis_dir+"y_prob_vs_all_srvf_pca_train_res.png")
plt.close()
print ("score平均斜率：",np.mean(np.abs(slopes)))
print ("score斜率标准差：",np.std(slopes))
slopes = np.array(slopes).reshape(-1,4)
print ("slopes.shape:", slopes.shape)
for i , tag in enumerate(param_dict.keys()):
    print (tag, "的支配性PC是:", np.max(np.abs(slopes[:,i])), np.argmax(np.abs(slopes[:,i])), "其平均绝对斜率是:",np.mean(np.abs(slopes[:,i])))


fig, axes = plt.subplots((PCA_N_COMPONENTS//8), 4, figsize=(15, 15))
for i in  range(PCA_N_COMPONENTS//8):
    for j in range(4):
        ax = axes[i][j]
        for tag in param_dict.keys():
            indices = [idx for idx, label in enumerate(quad_param_group) if label == tag]
            selected_data = np.array([all_srvf_pca.train_res[idx] for idx in indices])
            # prob_feature= np.max(y_prob[indices, :], axis=1)
            ax.scatter(selected_data[:,8*i+j],selected_data[:,8*i+j+1],color=colors[tag],alpha=0.6,label=tag)


# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_title('PC1 VS PC2')
# ax.legend()

plt.savefig(pca_anlysis_dir+"PC1_VS_PC2.png")
plt.close()



# ##############################
# # 绘制合成曲线的curvature和torsion
# U_synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(U_synthetic).reshape(sample_num, -1, 3)
# U_recovered = recovered_curves(U_synthetic_inverse, True)
# V_synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(V_synthetic).reshape(sample_num, -1, 3)
# V_recovered = recovered_curves(V_synthetic_inverse, True)
# C_synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(C_synthetic).reshape(sample_num, -1, 3)
# C_recovered = recovered_curves(C_synthetic_inverse, True)
# S_synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(S_synthetic).reshape(sample_num, -1, 3)
# S_recovered = recovered_curves(S_synthetic_inverse, True)
# # 使用函数绘制U, V, C等数据
# if sample_num < 6:
#     plot_recovered(U_recovered, U_curvatures, U_torsions, "U", weights, geometry_dir + "U_srvf_synthetic.png")
#     plot_recovered(V_recovered, V_curvatures, V_torsions, "V", weights, geometry_dir + "V_srvf_synthetic.png")
#     plot_recovered(C_recovered, C_curvatures, C_torsions, "C", weights, geometry_dir + "C_srvf_synthetic.png")
#     plot_recovered(S_recovered, S_curvatures, S_torsions, "S", weights, geometry_dir + "S_srvf_synthetic.png")
# else:
#     # 现在，为每一个类别调用新函数
#     # plot_stats_by_label(U_recovered, labels_U, U_curvatures, U_torsions, "U", weights, geometry_dir + "U")
#     # plot_stats_by_label(V_recovered, labels_V, V_curvatures, V_torsions, "V", weights, geometry_dir + "V")
#     # plot_stats_by_label(C_recovered, labels_C, C_curvatures, C_torsions, "C", weights, geometry_dir + "C")
#     # plot_stats_by_label(S_recovered, labels_S, S_curvatures, S_torsions, "S", weights, geometry_dir + "S")
#     plot_recovered_stats(U_recovered, U_curvatures, U_torsions, "U", weights, geometry_dir + "U_srvf_synthetic.png")
#     plot_recovered_stats(V_recovered, V_curvatures, V_torsions, "V", weights, geometry_dir + "V_srvf_synthetic.png")
#     plot_recovered_stats(C_recovered, C_curvatures, C_torsions, "C", weights, geometry_dir + "C_srvf_synthetic.png")
#     plot_recovered_stats(S_recovered, S_curvatures, S_torsions, "S", weights, geometry_dir + "C_srvf_synthetic.png")
#     # 调用上述函数为每个数据集绘制图形

# srvf_synthetics = np.concatenate([U_synthetic, V_synthetic, C_synthetic, S_synthetic], axis=0)
# print ("srvf_synthetics.shape: ", srvf_synthetics.shape)
# srvf_recovers = np.concatenate([U_recovered, V_recovered, C_recovered, S_recovered], axis=0)
# print ("srvf_recovers.shape: ", srvf_recovers.shape)


# C_srvf_synthetic_curvatures, C_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(C_recovered,weights)
# U_srvf_synthetic_curvatures, U_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(U_recovered,weights)
# V_srvf_synthetic_curvatures, V_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(V_recovered,weights)
# S_srvf_synthetic_curvatures, S_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(S_recovered,weights)

####################为SRVF PCA绘制violinplot####################


# 创建一个DataFrame
df = pd.DataFrame(all_srvf_pca.train_res, columns=[f'PC{i+1}' for i in range(PCA_N_COMPONENTS)])
df['Type'] = quad_param_group
# 创建一个4x4的子图网格
fig, axes = plt.subplots((PCA_N_COMPONENTS//Multi_plot_rows), Multi_plot_rows, figsize=(20, 20))
# 为每个主成分绘制violinplot
for i in range(PCA_N_COMPONENTS):
    ax = axes[i // Multi_plot_rows, i % Multi_plot_rows]
    sns.violinplot(x='Type', y=f'PC{i+1}', data=df, ax=ax, inner='quartile', palette=colors)  # inner='quartile' 在violin内部显示四分位数
    ax.set_title(f'Principal Component {i+1}')
    ax.set_ylabel('')  # 移除y轴标签，使得图更加简洁
plt.tight_layout()
plt.savefig(pca_anlysis_dir+"srvfPCA_total_Violinplot.png")
plt.close()

####################为srvf PCA绘制QQplot####################
# from numpy.polynomial.polynomial import Polynomial

# unique_types = df['Type'].unique()
# colors = sns.color_palette(n_colors=len(unique_types))

# # Create a 4x4 subplot
# fig, axes = plt.subplots(4, 4, figsize=(20, 20),dpi=300)
# fig.suptitle('QQ plots: Types vs. Total')

# # For each PC, conduct QQ plot analysis
# for pc in range(1, PCA_N_COMPONENTS+1):
#     q = np.linspace(0, 1, 101)[1:-1]  # From 1% to 99%
#     quantiles_total = np.percentile(df[f'PC{pc}'], q*100)
    
#     # Locate the position of the subplot
#     ax = axes[(pc-1)//4, (pc-1)%4]
    
#     # For each Type, create a QQ plot
#     for idx, type_value in enumerate(unique_types):
#         type_data = df[df['Type'] == type_value][f'PC{pc}']
#         quantiles_type = np.percentile(type_data, q*100)
        
#         # Scatter plot
#         ax.scatter(quantiles_total, quantiles_type, 
#                    color=colors[idx],
#                    marker="x",
#                    s=40)
        
#         # Fit a linear model and compute the least-squares error
#         p = Polynomial.fit(quantiles_total, quantiles_type, 1)
#         x_fit = np.linspace(min(quantiles_total), max(quantiles_total), 500)
#         y_fit = p(x_fit)
#         least_squares_error = np.sum((p(quantiles_total) - quantiles_type)**2)
        
#         # Plot the linear fit
#         ax.plot(x_fit, y_fit, color=colors[idx], label=f'Error: {least_squares_error:.2f}', linestyle='dotted', alpha=.5)
        
#         # Annotate with least-squares error
#         ax.annotate(f'Error {type_value}: {least_squares_error:.2f}', 
#                     xy=(0.05, 0.9 - idx*0.05), 
#                     xycoords='axes fraction', 
#                     color=colors[idx],
#                     fontsize=14)
    
#     # Add diagonal line
#     ax.plot([min(quantiles_total), max(quantiles_total)], 
#             [min(quantiles_total), max(quantiles_total)], 
#             linestyle='--',color="dimgray")
#     ax.set_title(f'PC{pc}')
#     ax.grid(True)

# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.savefig(pca_anlysis_dir + 'QQ_plots_combined_srvf.png')
# # plt.show()
# plt.close()


log.write("PCA standardization: {}\n".format(PCA_STANDARDIZATION))
print ("所有PCA的标准化状态：", PCA_STANDARDIZATION)
###########################################

##########################################
##### 计算geodesic并评价线性相关性

geodesic_dir = mkdir(bkup_dir, "geodesic")
# FrechetMean = compute_frechet_mean(Procrustes_curves)
system_name = platform.system()
if system_name == "Windows":
    FrechetMean = compute_frechet_mean(Procrustes_curves)
    np.save("./bkup/FrechetMean.npy", FrechetMean)

elif system_name == "Darwin":  # Mac OS的系统名称为'Darwin'
    if os.path.exists("./FrechetMean.npy"):
        FrechetMean = np.load("./bkup/FrechetMean.npy")
    else:
        # raise ValueError("File './FrechetMean.npy' does not exist!")
        FrechetMean = np.mean(Procrustes_curves, axis=0)
else:
    print(f"Unsupported operating system: {system_name}")

ArithmeticMean = np.mean(Procrustes_curves, axis=0)

end_time = datetime.now()
total_time = end_time - start_time
print(dir_formatted_time, "is done in", total_time.seconds, "seconds.")
log.close()
open_folder_in_explorer(bkup_dir)
