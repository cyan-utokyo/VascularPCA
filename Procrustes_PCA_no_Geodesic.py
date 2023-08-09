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
from myvtk.customize_pca import *
# from myvtk.make_fig import *
import shutil
import os
# from myvtk.dtw import *
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
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from minisom import MiniSom
from sklearn.neighbors import KernelDensity

SCALETO1 = False
log = open("./log.txt", "w")
# 获取当前时间
start_time = datetime.now()
smooth_scale = 0.01
# 将时间格式化为 'yymmddhhmmss' 格式
dir_formatted_time = start_time.strftime('%y-%m-%d-%H-%M-%S')
log.write("Start at: {}\n".format(dir_formatted_time))
bkup_dir = mkdir("./", "save_data_Procrustess")
bkup_dir = mkdir(bkup_dir, dir_formatted_time)
current_file_path = os.path.abspath(__file__)
current_file_name = os.path.basename(__file__)
backup_file_path = os.path.join(bkup_dir, current_file_name)
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
    print (filename)
    if filename in ill:
        print (filename, "is found in illcases.txt, skip")
        continue
    # print (filename)
    new_type_value = shapetype.loc[shapetype[0] == filename, 2].iloc[0]
    Typevalues.append(new_type_value)
    pt, Curv, Tors, Radius, Abscissas, ptns, ftangent, fnormal, fbinormal = GetMyVtk(pre_files[idx], frenet=1)
    Files.append(pre_files[idx])
    pt = pt-np.mean(pt,axis=0)
    unaligned_curves.append(pt)
    radii.append(Radius)
    sma_curv = np.convolve(Curv, weights, 'valid')
    Curvatures.append(sma_curv)
    sma_tors = np.convolve(Tors, weights, 'valid')
    Torsions.append(sma_tors)
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
fig = plt.figure(dpi=300, figsize=(10, 4))
def setup_axes(position):
    ax = fig.add_subplot(position)
    ax2 = ax.twinx()
    ax.set_ylim(0, 1)
    ax2.set_ylim(-1, 1)
    ax.grid(linestyle=":", alpha=0.5)
    ax.tick_params(axis='y', colors='red', labelsize=8)  # 设置y轴的颜色和字体大小
    ax2.tick_params(labelsize=8)  # 设置另一个y轴的字体大小
    ax.spines['left'].set_color('red')  # 设置y轴线的颜色
    return ax, ax2

ax1, ax1a = setup_axes(221)
ax2, ax2a = setup_axes(222)
ax3, ax3a = setup_axes(223)
ax4, ax4a = setup_axes(224)
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
def plot_with_errorbars(ax, ax2, curv_data, tors_data):
    mean_curv = np.mean(curv_data, axis=0)
    std_curv = np.std(curv_data, axis=0)
    mean_tors = np.mean(tors_data, axis=0)
    std_tors = np.std(tors_data, axis=0)

    ax.plot(mean_curv, color="r", linewidth=1)
    ax.fill_between(range(len(mean_curv)), mean_curv - std_curv, mean_curv + std_curv, color="r", alpha=0.2)
    ax2.plot(mean_tors, color="k", linestyle='--', linewidth=1)
    ax2.fill_between(range(len(mean_tors)), mean_tors - std_tors, mean_tors + std_tors, color="k", alpha=0.2)

plot_with_errorbars(ax1, ax1a, C_curvatures, C_torsions)
plot_with_errorbars(ax2, ax2a, S_curvatures, S_torsions)
plot_with_errorbars(ax3, ax3a, U_curvatures, U_torsions)
plot_with_errorbars(ax4, ax4a, V_curvatures, V_torsions)
ax1.set_title("C")
ax2.set_title("S")
ax3.set_title("U")
ax4.set_title("V")
plt.tight_layout()
plt.savefig(geometry_dir + "/Curvatures_Torsions.png")
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
log.write("Scaled all curves to one.\n")

# SRVF计算
Procs_srvf_curves = np.zeros_like(Procrustes_curves)
for i in range(len(Procrustes_curves)):
    Procs_srvf_curves[i] = calculate_srvf(Procrustes_curves[i])

makeVtkFile(bkup_dir+"mean_curve.vtk", np.mean(Procrustes_curves,axis=0),[],[] )
mean_srvf_inverse = inverse_srvf(np.mean(Procs_srvf_curves,axis=0),np.zeros(3))
makeVtkFile(bkup_dir+"mean_srvf.vtk", mean_srvf_inverse,[],[] )

# Geodesic计算
log.write("- Geodesic is not computed.\n")
# log.write("- Geodesic distance is computed by SRVR, this is the only way that makes sense.\n")
# Procrustes_geodesic_d = compute_geodesic_dist(Procrustes_curves)

pca_standardization = 1

# frechet_mean_srvf = compute_frechet_mean(Procs_srvf_curves)
# frechet_mean_srvf = frechet_mean_srvf / measure_length(frechet_mean_srvf)
# 保存数据
all_srvf_pca = PCAHandler(Procs_srvf_curves.reshape(len(Procs_srvf_curves),-1), None, 16, pca_standardization)
all_srvf_pca.PCA_training_and_test()
all_srvf_pca.compute_kde()
joblib.dump(all_srvf_pca.pca, bkup_dir + 'srvf_pca_model.pkl')
np.save(bkup_dir+"pca_model_filename.npy",Files )
all_pca = PCAHandler(Procrustes_curves.reshape(len(Procrustes_curves),-1), None, 16, pca_standardization)
all_pca.PCA_training_and_test()
all_pca.compute_kde()
joblib.dump(all_pca.pca, bkup_dir + 'pca_model.pkl')
np.save(bkup_dir+"not_std_curves.npy", all_pca.train_data)
np.save(bkup_dir+"not_std_srvf.npy", all_srvf_pca.train_data)
np.save(bkup_dir+"not_std_filenames.npy",Files)

pca_anlysis_dir = mkdir(bkup_dir, "pca_analysis")
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
sample_num = 5
U_synthetic = U_srvf_kde.sample(sample_num)
V_synthetic = V_srvf_kde.sample(sample_num)
C_synthetic = C_srvf_kde.sample(sample_num)
plt.scatter(all_srvf_pca.train_res[:,2], all_srvf_pca.train_res[:,3], c='k', s=50, marker="x")
plt.scatter(U_synthetic[:,2], U_synthetic[:,3], c='w', s=30, edgecolor='r',alpha=0.4)
plt.scatter(V_synthetic[:,2], V_synthetic[:,3], c='w', s=30, edgecolor='b',alpha=0.4)
plt.scatter(C_synthetic[:,2], C_synthetic[:,3], c='w', s=30, edgecolor='g',alpha=0.4)
plt.xlabel('PC3')
plt.ylabel('PC4')
plt.savefig(pca_anlysis_dir + "srvf_pca_synthetic.png")
plt.close()
U_synthetic_inverse = all_srvf_pca.inverse_transform_from_loadings(U_synthetic).reshape(sample_num, -1, 3)
U_recovered = recovered_curves(U_synthetic_inverse, True)
for i in range(len(U_recovered)):
    U_c, U_t = compute_curvature_and_torsion(U_recovered[i])
    plt.plot(U_c, label="Curvature")
    plt.plot(U_t, label="Torsion")
plt.show()

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
U_synthetic = U_kde.sample(5)
V_synthetic = V_kde.sample(5)
C_synthetic = C_kde.sample(5)
plt.scatter(all_pca.train_res[:,1], all_pca.train_res[:,2], c='k', s=50, marker="x")
plt.scatter(U_synthetic[:,1], U_synthetic[:,2], c='w', s=30, edgecolor='r',alpha=0.4)
plt.scatter(V_synthetic[:,1], V_synthetic[:,2], c='w', s=30, edgecolor='b',alpha=0.4)
plt.scatter(C_synthetic[:,1], C_synthetic[:,2], c='w', s=30, edgecolor='g',alpha=0.4)
plt.xlabel('PC2')
plt.ylabel('PC3')
plt.savefig(pca_anlysis_dir + "pca_synthetic.png")
plt.close()

# 为每个不同的字母分配一个唯一的数字
mapping = {letter: i for i, letter in enumerate(set(Typevalues))}
# 使用映射替换原始列表中的每个字母
numeric_lst = [mapping[letter] for letter in Typevalues]
# 定义你的颜色映射
default_palette = sns.color_palette()
cmap = ListedColormap(default_palette)
fig = plt.figure(dpi=300, figsize=(20, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# 定义颜色规范
boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5] # 根据你的数据调整
norm = BoundaryNorm(boundaries, cmap.N, clip=True)
# 创建散点图
sc1 = ax1.scatter(all_srvf_pca.train_res[:, 0], all_srvf_pca.train_res[:, 1], c=numeric_lst, cmap=cmap, norm=norm)
sc2 = ax2.scatter(all_pca.train_res[:, 0], all_pca.train_res[:, 1], c=numeric_lst, cmap=cmap, norm=norm)
# 添加注释
for i in range(len(Typevalues)):
    filename = Files[i].split("/")[-1][:-12]
    ax1.annotate(filename, (all_srvf_pca.train_res[i, 0], all_srvf_pca.train_res[i, 1]))
    ax2.annotate(filename, (all_pca.train_res[i, 0], all_pca.train_res[i, 1]))

# 获取Typevalues中的唯一值并进行排序
unique_values = sorted(list(set(Typevalues)))
# 创建一个颜色和标签的列表
colors = [cmap(norm(mapping[val])) for val in unique_values]
labels = unique_values
# 创建patch对象
patches = [Patch(color=colors[i], label=labels[i]) for i in range(len(unique_values))]
# 在每个子图上添加图例
for ax in [ax1, ax2]:
    ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(handles=patches)
plt.savefig(pca_anlysis_dir+"PCA_total.png")
plt.close()


####################为SRVF PCA绘制violinplot####################
# 创建一个DataFrame
df = pd.DataFrame(all_srvf_pca.train_res, columns=[f'PC{i+1}' for i in range(16)])
df['Type'] = Typevalues
# 创建一个4x4的子图网格
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
# 为每个主成分绘制violinplot
for i in range(16):
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
for pc in range(1, 17):  # PC1 to PC16
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
df = pd.DataFrame(all_pca.train_res, columns=[f'PC{i+1}' for i in range(16)])
df['Type'] = Typevalues
# 创建一个4x4的子图网格
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
# 为每个主成分绘制violinplot
for i in range(16):
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
for pc in range(1, 17):  # PC1 to PC16
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




log.write("PCA standardization: {}\n".format(pca_standardization))
print ("所有PCA的标准化状态：", pca_standardization)
#
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
        coord_PCAs.append(PCAHandler(train_data, test_data,standardization=pca_standardization))
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
            united_internal_PCAs.append(PCAHandler(train_data[:,:,i], test_data[:,:,i],standardization=pca_standardization))
            united_internal_PCAs[-1].PCA_training_and_test()
            train_res_temp, test_res_temp = united_internal_PCAs[-1].train_res, united_internal_PCAs[-1].test_res
            train_res.append(train_res_temp)
            test_res.append(test_res_temp)
        train_data = np.array(train_res).transpose(1,0,2).reshape(train_num, -1)
        test_data = np.array(test_res).transpose(1,0,2).reshape(test_num, -1)
        united_PCAs.append(PCAHandler(train_data, test_data, standardization=pca_standardization))
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
        param_PCAs.append(PCAHandler(train_data, test_data,standardization=pca_standardization))
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