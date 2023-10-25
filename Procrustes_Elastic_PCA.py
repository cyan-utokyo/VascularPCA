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

warnings.filterwarnings("ignore")

PCA_N_COMPONENTS = 16
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
plot_curves_with_arrows(1, 2, Procrustes_curves, 
                        Procs_srvf_curves, 
                        Typevalues, 
                        geometry_dir + "/Procrustes_curves_with_srvf.png")
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

bar_width = 0.05  # 使柱子更细
num_types = len(type_vals)
spacing = 0.7  # 增加每组柱子之间的距离
positions = [pos * spacing for pos in range(len(labels))]
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
# 每个 Typevalue 的柱子位置
for idx, tv in enumerate(type_vals):
    counts = [counter[label][tv] for label in labels[:-1]]
    counts.append(overall_counter[tv])  # 添加全体数据的计数
    ax.bar([pos + idx * bar_width for pos in positions], counts, width=bar_width, label=tv)


plt.legend(title="Typevalues")
plt.title("Param Group vs Typevalues")
plt.ylabel("Count")
plt.xlabel("Param Group")
plt.xticks([pos + bar_width * (num_types / 2) for pos in positions], labels)
plt.tight_layout()
plt.savefig(bkup_dir+"Param_Group_vs_Typevalues.png")
plt.close()


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
    label: plt.cm.jet(i/len(param_group_unique_labels)) for i, label in enumerate(param_group_unique_labels)
}


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12, stratify=y)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# 定义Random Forests分类器
rf_clf = RandomForestClassifier(n_estimators=100, random_state=12)  # n_estimators代表决策树的数量

# 训练分类器
rf_clf.fit(X_train_scaled, y_train)

# 预测
y_pred = rf_clf.predict(X_scaled)

print("Classification Report:")
print(classification_report(y, y_pred))

# 获取预测概率
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
               s=sizes_for_label*sizes_for_label*75)  # 我乘以50是为了放大点的大小，您可以根据需要调整这个值
    
    # 更新索引
    index += len(energies)

# 显示图形
ax.set_xlabel('Curvature Energy')
ax.set_ylabel('Torsion Energy')
ax.set_title('Energy Scatter Plot by Label')
ax.legend()
plt.savefig(bkup_dir+"Energy_Scatter_Plot_by_Label.png")
plt.close()


markers = ['o', 's']
linestyles = [':', '--']
fig, axes = plt.subplots(4, 4, dpi=300, figsize=(17, 15))
for i in range(4):
    for j in range(4):
        ax = axes[i][j]
        ax.tick_params(axis='both', which='major', labelsize=5)
        for tag in param_dict.keys():
            indices = [i for i, label in enumerate(quad_param_group) if label == tag]
            selected_data = np.array([all_srvf_pca.train_res[i] for i in indices])
            # print (tag, np.array(selected_data).shape)
            # print (np.array(param_dict[tag]["Energy"]).shape)
            for q in [0, 1]:
                ax.scatter(selected_data[:,4*i+j],np.array(param_dict[tag]["Energy"])[:,q],
                           color=colors[tag],
                           alpha=0.6, 
                           marker=markers[q],
                           s=15)
                model = np.polyfit(selected_data[:,4*i+j], np.array(param_dict[tag]["Energy"])[:,q], 1)
                predicted = np.poly1d(model)
                print ((selected_data[:,4*i+j]).shape)
                ax.plot(selected_data[:,4*i+j], 
                        predicted(selected_data[:,4*i+j]), 
                        color=colors[tag], 
                        linewidth=1.0, 
                        linestyle=linestyles[q])
            print ('----')

        # if j == 0:
        #     ax.set_ylabel(f'Row {i+1}')
        # if i == 3:
        #     ax.set_xlabel(f'Feature {4*i+j+1}')

# plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3))
plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.savefig(pca_anlysis_dir+"energyVSpcs.png")
plt.close()






# y_prob 和 all_srvf_pca.train_res 的相关性分析
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
# 遍历all_srvf_pca.train_res的每一个特征
for i in range(4):
    for j in range(4):
        ax = axes[i][j]
        
        # 遍历y_prob的每一个标签
        for m, label in enumerate(param_group_unique_labels):
            feature_data = all_srvf_pca.train_res[:, 4*i+j]
            prob_data = y_prob[:, m]
            
            # Filter data points where y value is less than 0.1
            # mask = prob_data >= 0.3
            # feature_data = feature_data[mask]
            # prob_data = prob_data[mask]
            # prob_data = np.power(prob_data, 2)
            
            # 画散点图
            ax.scatter(feature_data, prob_data, color=colors[label], alpha=0.6, label=label)
            # 需要多求几个PC，让n_components=0.95，然后从里面挑16个。
            
            # 线性拟合
            if len(feature_data) > 1:  # Ensure there are at least 2 points for linear regression
                model = np.polyfit(feature_data, prob_data, 1)
                predicted = np.poly1d(model)
                ax.plot(feature_data, predicted(feature_data), color=colors[label])
            
        if j == 0:
            ax.set_ylabel(f'Row {i+1}')
        if i == 3:
            ax.set_xlabel(f'Feature {4*i+j+1}')

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3))
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.grid(linestyle='--')
plt.savefig(pca_anlysis_dir+"y_prob_vs_all_srvf_pca_train_res.png")
plt.close()


# def fit_kde(data):
#     kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
#     kde.fit(data)
#     return kde
# # Extracting data based on type
# C_srvf_data = all_srvf_pca.train_res[Typevalues == 'C']
# U_srvf_data = all_srvf_pca.train_res[Typevalues == 'U']
# S_srvf_data = all_srvf_pca.train_res[Typevalues == 'S']
# V_srvf_data = all_srvf_pca.train_res[Typevalues == 'V']
# # Compute KDEs using sklearn's KernelDensity
# C_srvf_kde = fit_kde(C_srvf_data)
# U_srvf_kde = fit_kde(U_srvf_data)
# S_srvf_kde = fit_kde(S_srvf_data)
# V_srvf_kde = fit_kde(V_srvf_data)
# sample_num = 1000
# U_synthetic = U_srvf_kde.sample(sample_num)
# V_synthetic = V_srvf_kde.sample(sample_num)
# C_synthetic = C_srvf_kde.sample(sample_num)
# S_synthetic = S_srvf_kde.sample(sample_num)
# # 定义一个函数来计算elbow值


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
