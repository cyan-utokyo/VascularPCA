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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")
# warnings.filterwarnings("ignore", category=UserWarning, module="mygeodesic_plot")
# warnings.filterwarnings("ignore", category=RuntimeWarning, module="geometry")

PCA_N_COMPONENTS = 16
SCALETO1 = False
PCA_STANDARDIZATION = 1
RECONSTRUCT_WITH_SRVF = True
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

# plot_curve_with_peaks_name = geometry_dir + "peak_of_{}_and_{}.png".format(i,j)
# plot_curves_with_peaks(i, j, Procrustes_curves, Curvatures, Torsions, 
#                        plot_curve_with_peaks_name, axes=(1,2), distance=30)


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
plot_curves_with_arrows(1, 2, Procrustes_curves, Procs_srvf_curves, Typevalues, geometry_dir + "/Procrustes_curves_with_srvf.png")

#####

# frechet_mean_srvf = compute_frechet_mean(Procs_srvf_curves)
# frechet_mean_srvf = frechet_mean_srvf / measure_length(frechet_mean_srvf)
# 保存数据

PCA_weight = np.mean(pre_Curvatures, axis=0)


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


#################################
# geometric parameters
# 为每个不同的字母分配一个唯一的数字
mapping = {letter: i for i, letter in enumerate(set(Typevalues))}
# 使用映射替换原始列表中的每个字母
numeric_lst = [mapping[letter] for letter in Typevalues]
# print(numeric_lst)

########################################
# plot各种type的平均曲率和扭率
radii = np.array(radii)
pre_Curvatures = np.array(pre_Curvatures)
pre_Torsions = np.array(pre_Torsions)
pre_C_curvatures = []
pre_C_torsions = []
pre_S_curvatures = []
pre_S_torsions = []
pre_U_curvatures = []
pre_U_torsions = []
pre_V_curvatures = []
pre_V_torsions = []
C_curvatures = []
C_torsions = []
S_curvatures = []
S_torsions = []
U_curvatures = []
U_torsions = []
V_curvatures = []
V_torsions = []
log.write("RECONSTRUCT_WITH_SRVF:"+str(RECONSTRUCT_WITH_SRVF)+"\n")
if RECONSTRUCT_WITH_SRVF:
    OG_data_inverse = all_srvf_pca.inverse_transform_from_loadings(all_srvf_pca.train_res).reshape(len(all_srvf_pca.train_res), -1, 3)
    OG_data_inverse = recovered_curves(OG_data_inverse, RECONSTRUCT_WITH_SRVF)
else:
    OG_data_inverse = all_pca.inverse_transform_from_loadings(all_pca.train_res).reshape(len(all_pca.train_res), -1, 3)
    OG_data_inverse = recovered_curves(OG_data_inverse, RECONSTRUCT_WITH_SRVF)

geo_dist_OG_to_reverse = []
length_reverse = []

print ("OG_data_inverse[0].shape:", OG_data_inverse[0].shape)

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

for i in range(len(unaligned_curves)):
    if Typevalues[i] == "C":
        pre_C_curvatures.append(pre_Curvatures[i])
        pre_C_torsions.append(pre_Torsions[i])
        C_curvatures.append(Curvatures[i])
        C_torsions.append(Torsions[i])
    elif Typevalues[i] == "S":
        pre_S_curvatures.append(pre_Curvatures[i])
        pre_S_torsions.append(pre_Torsions[i])
        S_curvatures.append(Curvatures[i])
        S_torsions.append(Torsions[i])
    elif Typevalues[i] == "U":
        pre_U_curvatures.append(pre_Curvatures[i])
        pre_U_torsions.append(pre_Torsions[i])
        U_curvatures.append(Curvatures[i])
        U_torsions.append(Torsions[i])
    elif Typevalues[i] == "V":
        pre_V_curvatures.append(pre_Curvatures[i])
        pre_V_torsions.append(pre_Torsions[i])
        V_curvatures.append(Curvatures[i])
        V_torsions.append(Torsions[i])
pre_C_curvatures = np.array(pre_C_curvatures)
pre_C_torsions = np.array(pre_C_torsions)
pre_U_curvatures = np.array(pre_U_curvatures)
pre_U_torsions = np.array(pre_U_torsions)
pre_V_curvatures = np.array(pre_V_curvatures)
pre_V_torsions = np.array(pre_V_torsions)
pre_S_curvatures = np.array(pre_S_curvatures)
pre_S_torsions = np.array(pre_S_torsions)
C_curvatures = np.array(C_curvatures)
C_torsions = np.array(C_torsions)
U_curvatures = np.array(U_curvatures)
U_torsions = np.array(U_torsions)
V_curvatures = np.array(V_curvatures)
V_torsions = np.array(V_torsions)
S_curvatures = np.array(S_curvatures)
S_torsions = np.array(S_torsions)
print ("count CUVS: ")
print (len(pre_C_curvatures),len(pre_U_curvatures),len(pre_V_curvatures),len(pre_S_curvatures))


# 数据整合

C_data = np.hstack([C_curvatures, C_torsions])
U_data = np.hstack([U_curvatures, U_torsions])
S_data = np.hstack([S_curvatures, S_torsions])
V_data = np.hstack([V_curvatures, V_torsions])


X = np.vstack([C_data, U_data, S_data, V_data])
y = np.array(['C']*len(C_data) + ['U']*len(U_data) + ['S']*len(S_data) + ['V']*len(V_data))

# 划分训练集和测试集，按照各类型内的数量比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用RFECV进行特征选择
class_weight = "balanced"
estimator = SVC(kernel="linear", class_weight=class_weight)
# 使用StratifiedKFold进行交叉验证，确保每个子集中各类别的样本比例与完整数据集中的相同
cv = StratifiedKFold(5)
selector = RFECV(estimator, step=2, cv=cv)
selector = selector.fit(X_train_scaled, y_train)
X_train_selected = X_train_scaled[:, selector.support_]
X_test_selected = X_test_scaled[:, selector.support_]
selected_indices = np.where(selector.support_)[0]
log.write("Feature selection result:"+str(selected_indices)+"\n")
# https://chat.openai.com/c/6a8aac2a-a04b-4b10-ae1b-869167a76263
# 使用所选特征初始化SVM模型
svm_classifier = SVC(probability=True, class_weight=class_weight)

# 使用cross-validation进行模型评估
scores = cross_val_score(svm_classifier, X_train_selected, y_train, cv=3)
log.write("Cross-validation scores after feature selection:"+str(scores)+"\n")
log.write("Average score after feature selection:"+str(np.mean(scores))+"\n")

# 训练模型
svm_classifier.fit(X_train_selected, y_train)

def score_data(data_point, model):
    # 使用模型预测
    probabilities = model.predict_proba([data_point])
    class_labels = model.classes_
    
    # 返回每个类别的概率
    scores = dict(zip(class_labels, probabilities[0]))
    return scores

# for i in range(len(X_test_selected)):
#     data_sample_selected = X_test_selected[i]
#     actual_label = y_test[i]
#     # 打印实际标签
#     log.write("Actual label:"+actual_label+"\n")
#     scores = score_data(data_sample_selected, svm_classifier)
#     log.write(str(scores)+"\n")

# 获取测试集的预测概率
y_prob = svm_classifier.predict_proba(X_test_selected)

# 为每个类别计算ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

# 获取类别的真实标签和预测概率
for i, label in enumerate(svm_classifier.classes_):
    true_label = np.where(y_test == label, 1, 0)
    prob = y_prob[:, i]
    fpr[label], tpr[label], _ = roc_curve(true_label, prob)
    roc_auc[label] = auc(fpr[label], tpr[label])

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
for label in svm_classifier.classes_:
    plt.plot(fpr[label], tpr[label], label=f'ROC curve of class {label} (area = {roc_auc[label]:.2f})')
    
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multi-class')
plt.legend(loc="lower right")
# plt.show()
plt.savefig(geometry_dir+"ROC_Curves_for_Multi-class.png")
plt.close()








fig = plt.figure(dpi=300,figsize=(9,6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i in range(len(C_curvatures)):
    ax1.plot(C_curvatures[i],color="C0",alpha=0.5)
    ax2.plot(C_torsions[i],color="C0",alpha=0.5)
for i in range(len(U_curvatures)):
    ax1.plot(U_curvatures[i],color="C1",alpha=0.5)
    ax2.plot(U_torsions[i],color="C1",alpha=0.5)
for i in range(len(S_curvatures)):
    ax1.plot(S_curvatures[i],color="C2",alpha=0.5)
    ax2.plot(S_torsions[i],color="C2",alpha=0.5)
for i in range(len(V_curvatures)):
    ax1.plot(V_curvatures[i],color="C3",alpha=0.8, linestyle="--")
    ax2.plot(V_torsions[i],color="C3",alpha=0.8, linestyle="--")
ax1.set_title("Curvatures")
ax2.set_title("Torsions")
plt.savefig(geometry_dir+"/tanh_conv_Curvatures_and_Torsions.png")
plt.close()

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
ax1 = setup_axes(221, 0, 0.75)
ax2 = setup_axes(222, 0, 0.75)
ax3 = setup_axes(223, 0, 0.75)
ax4 = setup_axes(224, 0, 0.75)
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

#################################
# 绘制每条血管内的曲率和扭率对比全体的偏离程度的散点图

fig = plt.figure(dpi=300, figsize=(9, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# 计算所有数据的特征点均值的均值
average_of_means_curvatures = np.mean([np.mean(curv) for curv in Curvatures])
average_of_means_torsions = np.mean([np.mean(tors) for tors in Torsions])

param_group = []
for i in range(len(Curvatures)):
    if np.mean(Curvatures[i]) > average_of_means_curvatures and np.mean(Torsions[i]) > average_of_means_torsions:
        param_group.append('高曲率+高扭曲')
        ax1.scatter(np.mean(Curvatures, axis=0), Curvatures[i], marker="o", color="red")
        ax2.scatter(np.mean(Torsions, axis=0), Torsions[i], marker="o", color="red")
    elif np.mean(Curvatures[i]) > average_of_means_curvatures and np.mean(Torsions[i]) <= average_of_means_torsions:
        param_group.append('高曲率+低扭曲')
        ax1.scatter(np.mean(Curvatures, axis=0), Curvatures[i], marker="o", color="green")
        ax2.scatter(np.mean(Torsions, axis=0), Torsions[i], marker="o", color="green")
    elif np.mean(Curvatures[i]) <= average_of_means_curvatures and np.mean(Torsions[i]) > average_of_means_torsions:
        param_group.append('低曲率+高扭曲')
        ax1.scatter(np.mean(Curvatures, axis=0), Curvatures[i], marker="o", color="blue")
        ax2.scatter(np.mean(Torsions, axis=0), Torsions[i], marker="o", color="blue")
    else:
        param_group.append('低曲率+低扭曲')
        ax1.scatter(np.mean(Curvatures, axis=0), Curvatures[i], marker="o", color="orange")
        ax2.scatter(np.mean(Torsions, axis=0), Torsions[i], marker="o", color="orange")
    print (param_group[-1], Typevalues[i])



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
for ax in [ax1, ax2]:
    ax.grid(linestyle=":", alpha=0.5)
plt.savefig(geometry_dir + "/case_param_compare.png")
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

# fig, axes = plt.subplots(4, 2, dpi=300, figsize=(10, 8))
fig, axes = plt.subplots(1, 2, dpi=300, figsize=(8, 3))
def plot_with_shade(ax, data_samples,  title, ymin, ymax, color="red"):
    if len(data_samples.shape) == 3:  # bootstrap samples
        x = range(data_samples.shape[2])
    else:  # means or stds
        x = range(data_samples.shape[0])
    # x = range(data_samples.shape[2])
    # 定义箱型图样式参数
    box_properties = {
        'color': color,
        'linewidth': 1
    }
    whisker_properties = {
        'color': color,
        'linewidth': 1
    }
    cap_properties = {
        'color': color,
        'linewidth': 1
    }
    median_properties = {
        'color': color,
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

plot_with_shade(axes[0], compute_bootstrap_statistics(bootstrap_samples_C_curvature)[0], "Means", 0, 0.6, color="blue")
plot_with_shade(axes[1], compute_bootstrap_statistics(bootstrap_samples_C_curvature)[1], "Stds", 0, 0.4, color="blue")
plot_with_shade(axes[0], compute_bootstrap_statistics(bootstrap_samples_S_curvature)[0], "Means", 0, 0.6, color="orange")
plot_with_shade(axes[1], compute_bootstrap_statistics(bootstrap_samples_S_curvature)[1], "Stds", 0, 0.4, color="orange")
plot_with_shade(axes[0], compute_bootstrap_statistics(bootstrap_samples_U_curvature)[0], "Means", 0, 0.6, color="red")
plot_with_shade(axes[1], compute_bootstrap_statistics(bootstrap_samples_U_curvature)[1], "Stds", 0, 0.4, color="red")
plot_with_shade(axes[0], compute_bootstrap_statistics(bootstrap_samples_V_curvature)[0], "Means", 0, 0.6, color="green")
plot_with_shade(axes[1], compute_bootstrap_statistics(bootstrap_samples_V_curvature)[1], "Stds", 0, 0.4, color="green")


plt.tight_layout()
plt.savefig(geometry_dir+"Bootstrap_Distributions_with_Global_Curvature.png")
plt.close()


fig, axes = plt.subplots(1, 2, dpi=300, figsize=(8, 3))
plot_with_shade(axes[0], compute_bootstrap_statistics(bootstrap_samples_C_torsion)[0], "Means", -0.6, 0.6, color="blue")
plot_with_shade(axes[1], compute_bootstrap_statistics(bootstrap_samples_C_torsion)[1], "Stds", 0, 1.2, color="blue")
plot_with_shade(axes[0], compute_bootstrap_statistics(bootstrap_samples_S_torsion)[0], "Means", -0.6, 0.6, color="orange")
plot_with_shade(axes[1], compute_bootstrap_statistics(bootstrap_samples_S_torsion)[1], "Stds", 0, 1.2, color="orange")
plot_with_shade(axes[0], compute_bootstrap_statistics(bootstrap_samples_U_torsion)[0], "Means", -0.6, 0.6, color="red")
plot_with_shade(axes[1], compute_bootstrap_statistics(bootstrap_samples_U_torsion)[1], "Stds", 0, 1.2, color="red")
plot_with_shade(axes[0], compute_bootstrap_statistics(bootstrap_samples_V_torsion)[0], "Means", -0.6, 0.6, color="green")
plot_with_shade(axes[1], compute_bootstrap_statistics(bootstrap_samples_V_torsion)[1], "Stds", 0, 1.2, color="green")
plt.tight_layout()
plt.savefig(geometry_dir+"Bootstrap_Distributions_with_Global_Torsion.png")
plt.close()


# geometric parameters
#################################

C_curvatures_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(C_curvatures)
U_curvatures_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(U_curvatures)
S_curvatures_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(S_curvatures)
V_curvatures_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(V_curvatures)
C_torsions_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(C_torsions)
U_torsions_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(U_torsions)
S_torsions_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(S_torsions)
V_torsions_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(V_torsions)

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


C_srvf_synthetic_curvatures, C_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(C_recovered,weights)
U_srvf_synthetic_curvatures, U_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(U_recovered,weights)
V_srvf_synthetic_curvatures, V_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(V_recovered,weights)
S_srvf_synthetic_curvatures, S_srvf_synthetic_torsions = compute_synthetic_curvature_and_torsion(S_recovered,weights)

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

C_synthetic_curvatures, C_synthetic_torsions = compute_synthetic_curvature_and_torsion(C_recovered,weights)
U_synthetic_curvatures, U_synthetic_torsions = compute_synthetic_curvature_and_torsion(U_recovered,weights)
V_synthetic_curvatures, V_synthetic_torsions = compute_synthetic_curvature_and_torsion(V_recovered,weights)
S_synthetic_curvatures, S_synthetic_torsions = compute_synthetic_curvature_and_torsion(S_recovered,weights)

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
from numpy.polynomial.polynomial import Polynomial

unique_types = df['Type'].unique()
colors = sns.color_palette(n_colors=len(unique_types))

# Create a 4x4 subplot
fig, axes = plt.subplots(4, 4, figsize=(20, 20),dpi=300)
fig.suptitle('QQ plots: Types vs. Total')

# For each PC, conduct QQ plot analysis
for pc in range(1, PCA_N_COMPONENTS+1):
    q = np.linspace(0, 1, 101)[1:-1]  # From 1% to 99%
    quantiles_total = np.percentile(df[f'PC{pc}'], q*100)
    
    # Locate the position of the subplot
    ax = axes[(pc-1)//4, (pc-1)%4]
    
    # For each Type, create a QQ plot
    for idx, type_value in enumerate(unique_types):
        type_data = df[df['Type'] == type_value][f'PC{pc}']
        quantiles_type = np.percentile(type_data, q*100)
        
        # Scatter plot
        ax.scatter(quantiles_total, quantiles_type, 
                   color=colors[idx],
                   marker="x",
                   s=40)
        
        # Fit a linear model and compute the least-squares error
        p = Polynomial.fit(quantiles_total, quantiles_type, 1)
        x_fit = np.linspace(min(quantiles_total), max(quantiles_total), 500)
        y_fit = p(x_fit)
        least_squares_error = np.sum((p(quantiles_total) - quantiles_type)**2)
        
        # Plot the linear fit
        ax.plot(x_fit, y_fit, color=colors[idx], label=f'Error: {least_squares_error:.2f}', linestyle='dotted', alpha=.5)
        
        # Annotate with least-squares error
        ax.annotate(f'Error {type_value}: {least_squares_error:.2f}', 
                    xy=(0.05, 0.9 - idx*0.05), 
                    xycoords='axes fraction', 
                    color=colors[idx],
                    fontsize=14)
    
    # Add diagonal line
    ax.plot([min(quantiles_total), max(quantiles_total)], 
            [min(quantiles_total), max(quantiles_total)], 
            linestyle='--',color="dimgray")
    ax.set_title(f'PC{pc}')
    ax.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig(pca_anlysis_dir + 'QQ_plots_combined_srvf.png')
# plt.show()
plt.close()


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
# 对于每个PC，进行QQ图分析
# 创建4x4的子图大图
unique_types = df['Type'].unique()
colors = sns.color_palette(n_colors=len(unique_types))

# Create a 4x4 subplot
fig, axes = plt.subplots(4, 4, figsize=(20, 20),dpi=300)
fig.suptitle('QQ plots: Types vs. Total')

# For each PC, conduct QQ plot analysis
for pc in range(1, PCA_N_COMPONENTS+1):
    q = np.linspace(0, 1, 101)[1:-1]  # From 1% to 99%
    quantiles_total = np.percentile(df[f'PC{pc}'], q*100)
    
    # Locate the position of the subplot
    ax = axes[(pc-1)//4, (pc-1)%4]
    
    # For each Type, create a QQ plot
    for idx, type_value in enumerate(unique_types):
        type_data = df[df['Type'] == type_value][f'PC{pc}']
        quantiles_type = np.percentile(type_data, q*100)
        
        # Scatter plot
        ax.scatter(quantiles_total, quantiles_type, 
                   color=colors[idx],
                   marker="x",
                   s=40)
        
        # Fit a linear model and compute the least-squares error
        p = Polynomial.fit(quantiles_total, quantiles_type, 1)
        x_fit = np.linspace(min(quantiles_total), max(quantiles_total), 500)
        y_fit = p(x_fit)
        least_squares_error = np.sum((p(quantiles_total) - quantiles_type)**2)
        
        # Plot the linear fit
        ax.plot(x_fit, y_fit, color=colors[idx], label=f'Error: {least_squares_error:.2f}', linestyle='dotted', alpha=.5)
        
        # Annotate with least-squares error
        ax.annotate(f'Error {type_value}: {least_squares_error:.2f}', 
                    xy=(0.05, 0.9 - idx*0.05), 
                    xycoords='axes fraction', 
                    color=colors[idx],
                    fontsize=14)
    
    # Add diagonal line
    ax.plot([min(quantiles_total), max(quantiles_total)], 
            [min(quantiles_total), max(quantiles_total)], 
            linestyle='--',color="dimgray")
    ax.set_title(f'PC{pc}')
    ax.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig(pca_anlysis_dir + 'QQ_plots_combined.png')
# plt.show()
plt.close()


log.write("PCA standardization: {}\n".format(PCA_STANDARDIZATION))
print ("所有PCA的标准化状态：", PCA_STANDARDIZATION)
###########################################

###########################################
# 找出curvature和torsion分布最典型的曲线
curvature_kdes = [U_curvatures_kde, V_curvatures_kde, C_curvatures_kde, S_curvatures_kde]
torsion_kdes = [U_torsions_kde, V_torsions_kde, C_torsions_kde, S_torsions_kde]
def compute_likelihood(kde, data):
    log_likelihood = kde.score_samples(data)
    return np.exp(log_likelihood)

def get_top_5_indices_for_label(curvatures, torsions, curvature_kde, torsion_kde):
    curv_likelihoods = compute_likelihood(curvature_kde, curvatures)
    tors_likelihoods = compute_likelihood(torsion_kde, torsions)
    
    combined_likelihoods = curv_likelihoods * tors_likelihoods
    
    # Get the indices of the top 5 likelihood values
    top_5_indices = combined_likelihoods.argsort()[-5:][::-1]
    return top_5_indices

# Initialize result dictionary
typical_param_results = {}

curv_likelihoods_data = {}
tors_likelihoods_data = {}
# Iterate over each unique label and compute likelihoods
labels = ["U", "V", "C", "S"]
for i, label in enumerate(labels):
    indices_for_label = np.where(Typevalues == label)[0]
    curvatures_for_label = Curvatures[indices_for_label]
    torsions_for_label = Torsions[indices_for_label]
    
    curv_likelihoods = compute_likelihood(curvature_kdes[i], curvatures_for_label)
    tors_likelihoods = compute_likelihood(torsion_kdes[i], torsions_for_label)
    
    curv_likelihoods_data[label] = curv_likelihoods
    tors_likelihoods_data[label] = tors_likelihoods 
      
    top_5_indices = get_top_5_indices_for_label(curvatures_for_label, 
                                               torsions_for_label, 
                                               curvature_kdes[i], 
                                               torsion_kdes[i])
    
    # Map the local indices back to global indices
    global_indices = indices_for_label[top_5_indices]
    typical_param_results[label] = global_indices

xtick_labels = [f"{label} Curvature" for label in labels] + [f"{label} Torsion" for label in labels]
# Plotting boxplots using seaborn
plt.figure(figsize=(12, 6))
sns.boxplot(data=[curv_likelihoods_data[label] for label in labels] + 
                 [tors_likelihoods_data[label] for label in labels], 
            orient='v')
plt.xticks(list(range(8)), xtick_labels, rotation=45)  # Rotate the xtick labels for better readability
plt.xlabel('Labels')
plt.ylabel('Likelihoods')
plt.title('Boxplot of curv_likelihoods and tors_likelihoods')
plt.tight_layout()  # Ensure all labels fit well within the figure
plt.savefig(geometry_dir + "curv_tors_likelihoods_boxplot.png")
plt.close()

# 找出curvature和torsion分布最典型的曲线
###########################################

##########################################
##### 计算geodesic并评价线性相关性

geodesic_dir = mkdir(bkup_dir, "geodesic")
# U, U, V, V, C, C, S, S
# case_study = ['BG11_Left', 'BG18_Left',
#               'BH0031_Left', 'BH0017_Right',
#               'BH0006_Left', 'BH0003_Right',
#               'BH0012_Left', 'BH0016_Left']
# for i in range(len(Files)):
#     for c in case_study:
#         if c in Files[i]:
#             spec.append(Procrustes_curves[i])
spec_label = ['U1','U2',
              'V1','V2',
              'C1','C2',
              'S1','S2',
              'FrechetMean', 'ArithmeticMean']

pca_kdes = [U_kde, V_kde, C_kde, S_kde]
srvf_pca_kdes = [U_srvf_kde, V_srvf_kde, C_srvf_kde, S_srvf_kde]

case_study = []
spec = []
for key, values in typical_param_results.items():
    print(key, ":", values[:2])
    case_study.extend(values[:2])
    spec.append(Procrustes_curves[values[0]])
    spec.append(Procrustes_curves[values[1]])


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
spec.append(FrechetMean)
spec.append(ArithmeticMean)
spec = np.array(spec)
print ("spec.shape: ", spec.shape)
print ("spec_label: ", len(spec_label))

for j in [0, 1]:
    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        if i == j:
            continue
        spec2mean_dir = mkdir(geodesic_dir, "{}_2_{}".format(spec_label[j], spec_label[i]))
        curve_a = spec[j]
        curve_b = spec[i]
        geodesic_dist_a2b = compute_geodesic_dist_between_two_curves(curve_a, curve_b)
        # num_step = int(geodesic_dist_a2b/0.5)
        num_step = 10
        shapes_along_geodesic  = compute_geodesic_shapes_between_two_curves(curve_a, curve_b, num_steps=num_step)
        shapes_along_geodesic = np.vstack(([curve_a], shapes_along_geodesic, [curve_b]))
        shape_srvf_along_geodesic = []
        for k in range(len(shapes_along_geodesic)):
            makeVtkFile(spec2mean_dir+"shape{}.vtk".format(k), shapes_along_geodesic[k], [],[])
            shape_srvf_along_geodesic.append(calculate_srvf(shapes_along_geodesic[k]))
        shape_srvf_along_geodesic = np.array(shape_srvf_along_geodesic)
        spec_geo_res = all_pca.pca.transform((shapes_along_geodesic.reshape(len(shapes_along_geodesic),-1)-all_pca.train_mean)/all_pca.train_std)
        spec_geo_srvf_res = all_srvf_pca.pca.transform((shape_srvf_along_geodesic.reshape(len(shape_srvf_along_geodesic),-1)-all_srvf_pca.train_mean)/all_srvf_pca.train_std)
        pca_likelihood = []
        for k in range(len(shapes_along_geodesic)):
            seed_pca_density = pca_kdes[j//2].score_samples(spec_geo_res[k].reshape(1, -1))
            seed_pca_likelihood = np.exp(seed_pca_density)
            target_pca_density = pca_kdes[i//2].score_samples(spec_geo_res[k].reshape(1, -1))
            target_pca_likelihood = np.exp(target_pca_density)
            seed_srvf_pca_density = srvf_pca_kdes[j//2].score_samples(spec_geo_srvf_res[k].reshape(1, -1))
            seed_srvf_pca_likelihood = np.exp(seed_srvf_pca_density)
            target_srvf_pca_density = srvf_pca_kdes[i//2].score_samples(spec_geo_srvf_res[k].reshape(1, -1))
            target_srvf_pca_likelihood = np.exp(target_srvf_pca_density)
            pca_likelihood.append([seed_pca_likelihood, target_pca_likelihood, seed_srvf_pca_likelihood, target_srvf_pca_likelihood])
        pca_likelihood = np.array(pca_likelihood)
        fig = plt.figure(dpi=300, figsize=(6, 5))
        ax1 = fig.add_subplot(111)
        ax1.plot(pca_likelihood[:, 0], linestyle='solid', color="k",label="seed_pca")
        ax1.plot(pca_likelihood[:, 1], linestyle='solid', color="r",label="target_pca")
        ax1.plot(pca_likelihood[:, 2], linestyle='dashed', color="k", label="seed_srvf_pca")
        ax1.plot(pca_likelihood[:, 3], linestyle="dashed", color="r",label="target_srvf_pca")
        ax1.legend()
        ax1.grid(linestyle='--', linewidth=0.5)
        plt.savefig(spec2mean_dir+"pca_likelihood.png")
        plt.close()

        # 计算曲率和扭率
        geod_curvature = []
        geod_torsion = []
        shape_likelihoods = []
        for k in range(len(shapes_along_geodesic)):
            spec_geo_curvatures, spec_geo_torsions = compute_curvature_and_torsion(shapes_along_geodesic[k])
            sma_curv = np.convolve(spec_geo_curvatures, weights, 'valid')
            geod_curvature.append(sma_curv)
            sma_tors = np.convolve(spec_geo_torsions, weights, 'valid')
            geod_torsion.append(sma_tors)
            seed_curvature_log_density = curvature_kdes[j//2].score_samples(sma_curv.reshape(1, -1))
            seed_curvature_density = np.exp(seed_curvature_log_density)
            target_curvature_log_density = curvature_kdes[i//2].score_samples(sma_curv.reshape(1, -1))
            target_curvature_density = np.exp(target_curvature_log_density)
            seed_torsion_log_density = torsion_kdes[j//2].score_samples(sma_tors.reshape(1, -1))
            seed_torsion_density = np.exp(seed_torsion_log_density)
            target_torsion_log_density = torsion_kdes[i//2].score_samples(sma_tors.reshape(1, -1))
            target_torsion_density = np.exp(target_torsion_log_density)
            shape_likelihoods.append([seed_curvature_log_density, target_curvature_log_density, seed_torsion_log_density, target_torsion_log_density])
        geod_curvature = np.array(geod_curvature)
        geod_torsion = np.array(geod_torsion)
        shape_likelihoods = np.array(shape_likelihoods)
        plot_curvature_torsion_heatmaps(geod_curvature, geod_torsion, spec2mean_dir+"geo_curvature_torsion.png")
        fig = plt.figure(dpi=300, figsize=(6, 5))
        ax1 = fig.add_subplot(111)
        ax1.plot(shape_likelihoods[:, 0], linestyle='solid', color="k",label="seed_curvature")
        ax1.plot(shape_likelihoods[:, 1], linestyle='solid', color="r",label="target_curvature")
        ax1.plot(shape_likelihoods[:, 2], linestyle='dashed', color="k", label="seed_torsion")
        ax1.plot(shape_likelihoods[:, 3], linestyle="dashed", color="r",label="target_torsion")
        ax1.legend()
        ax1.grid(linestyle='--', linewidth=0.5)
        plt.savefig(spec2mean_dir+"likelihood.png")
        plt.close()

        # 首先生成颜色映射
        geod_cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
        # 生成颜色值数组
        # n_points = num_step
        geod_colors = geod_cmap(np.linspace(0, 1, len(shapes_along_geodesic)))
        plot_curves_on_2d(curve_a, curve_b, shapes_along_geodesic, 
                        spec2mean_dir+"geo_deformation.png")

        fig = plt.figure(dpi=300, figsize=(6, 5))
        ax1 = fig.add_subplot(111)
        ax1.scatter(all_srvf_pca.train_res[:, 0], all_srvf_pca.train_res[:, 1], c="white", alpha=0.3, edgecolors='dimgray')
        ax1.scatter(spec_geo_srvf_res[1:-1, 0], spec_geo_srvf_res[1:-1, 1], c=geod_colors[1:-1], alpha=0.9, label='Shapes Along Geod')
        ax1.scatter(spec_geo_srvf_res[0, 0], spec_geo_srvf_res[0, 1], c="k", alpha=0.9, label='Seed')
        ax1.scatter(spec_geo_srvf_res[-1, 0], spec_geo_srvf_res[-1, 1], c="r", alpha=0.9, label='Target')
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.legend(loc='lower right')
        plt.savefig(spec2mean_dir+"srvf_PCA_total_with_geo.png")
        plt.close()
        fig = plt.figure(dpi=300, figsize=(6, 5))
        ax2 = fig.add_subplot(111)
        ax2.scatter(all_pca.train_res[:, 0], all_pca.train_res[:, 1], c="white", alpha=0.3, edgecolors='dimgray')
        ax2.scatter(spec_geo_res[1:-1, 0], spec_geo_res[1:-1, 1], c=geod_colors[1:-1], alpha=0.9, label='Shapes Along Geod')
        ax2.scatter(spec_geo_res[0, 0], spec_geo_res[0, 1], c="k", alpha=0.9, label='Seed')
        ax2.scatter(spec_geo_res[-1, 0], spec_geo_res[-1, 1], c="r", alpha=0.9, label='Target')
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.grid(linestyle='--', linewidth=0.5)
        plt.legend(loc='lower right')
        plt.savefig(spec2mean_dir+"PCA_total_with_geo.png")
        plt.close()

#######################################
# 每个type的最典形状在PCA空间中的分布
spec_srvf = []
for i in range(len(spec)):
    spec_srvf.append(calculate_srvf(spec[i]))
spec_srvf = np.array(spec_srvf)
spec_srvf_res = all_srvf_pca.pca.transform((spec_srvf.reshape(len(spec_srvf),-1)-all_srvf_pca.train_mean)/all_srvf_pca.train_std)
fig = plt.figure(dpi=300, figsize=(6, 5))
ax1 = fig.add_subplot(111)
ax1.scatter(all_srvf_pca.train_res[:, 0], all_srvf_pca.train_res[:, 1], c="white", alpha=0.7, edgecolors='dimgray')
ax1.scatter(spec_srvf_res[-1, 0], spec_srvf_res[-1, 1], c="r", alpha=0.9, marker='*', label='ArithmeticMean')
ax1.scatter(spec_srvf_res[-2, 0], spec_srvf_res[-2, 1], c="dimgray", alpha=0.9, marker='*', label='FrechetMean')
ax1.scatter(spec_srvf_res[:2, 0], spec_srvf_res[:2, 1], c="r", alpha=0.7, edgecolors='white', label='U')
ax1.scatter(spec_srvf_res[2:4, 0], spec_srvf_res[2:4, 1], c="g", alpha=0.7, edgecolors='white', label='V')
ax1.scatter(spec_srvf_res[4:6, 0], spec_srvf_res[4:6, 1], c="b", alpha=0.7, edgecolors='white', label='C')
ax1.scatter(spec_srvf_res[6:8, 0], spec_srvf_res[6:8, 1], c="orange", alpha=0.7, edgecolors='white', label='S')
ax1.grid(linestyle='--', linewidth=0.5)
plt.legend(loc='upper center')
plt.savefig(geodesic_dir+"srvf_PCA_total_with_spec.png")
plt.close()
spec_res = all_pca.pca.transform((spec.reshape(len(spec),-1)-all_pca.train_mean)/all_pca.train_std)
fig = plt.figure(dpi=300, figsize=(6, 5))
ax2 = fig.add_subplot(111)
ax2.scatter(all_pca.train_res[:, 0], all_pca.train_res[:, 1], c="white", alpha=0.7, edgecolors='dimgray')
ax2.scatter(spec_res[-1, 0], spec_res[-1, 1], c="r", alpha=0.9, marker='*', label='ArithmeticMean')
ax2.scatter(spec_res[-2, 0], spec_res[-2, 1], c="dimgray", alpha=0.9, marker='*', label='FrechetMean')
ax2.scatter(spec_res[:2, 0], spec_res[:2, 1], c="r", alpha=0.7, edgecolors='white', label='U')
ax2.scatter(spec_res[2:4, 0], spec_res[2:4, 1], c="g", alpha=0.7, edgecolors='white', label='V')
ax2.scatter(spec_res[4:6, 0], spec_res[4:6, 1], c="b", alpha=0.7, edgecolors='white', label='C')
ax2.scatter(spec_res[6:8, 0], spec_res[6:8, 1], c="orange", alpha=0.7, edgecolors='white', label='S')
ax2.grid(linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig(geodesic_dir+"PCA_total_with_spec.png")
plt.close()
# 每个type的最典形状在PCA空间中的分布
#######################################

####################################################
# 计算所有曲线与FrechetMean和ArithmeticMean的测地线距离
frechet_distances = [compute_geodesic_dist_between_two_curves(curve, FrechetMean) for curve in Procrustes_curves]
arithmetic_distances = [compute_geodesic_dist_between_two_curves(curve, ArithmeticMean) for curve in Procrustes_curves]
# 计算spec[:8]与Procrustes_curves中每条曲线的测地线距离
spec_distances = [[compute_geodesic_dist_between_two_curves(s_curve, p_curve) for p_curve in Procrustes_curves] for s_curve in spec[:8]]
# 整理数据为一个列表
all_distances = [frechet_distances, arithmetic_distances] + spec_distances
# 设定标签
labels = ['Frechet Mean', 'Arithmetic Mean'] + spec_label[:8]

# 使用boxplot展示结果
fig = plt.figure(dpi=300, figsize=(6, 5))
ax = fig.add_subplot(111)
ax.boxplot(all_distances, vert=True, patch_artist=True, labels=labels)
ax.set_ylabel('Geodesic Distance')
ax.set_title('Boxplot of Geodesic Distances to Means and Spec')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(geodesic_dir+"boxplot_of_geodesic_distances_to_means_and_spec.png")
plt.close()
# 计算所有曲线与FrechetMean和ArithmeticMean的测地线距离
####################################################

C_synthetic_params = np.concatenate((C_synthetic_curvatures, C_synthetic_torsions), axis=1) # 长度为1000
U_synthetic_params = np.concatenate((U_synthetic_curvatures, U_synthetic_torsions), axis=1) # 长度为1000
V_synthetic_params = np.concatenate((V_synthetic_curvatures, V_synthetic_torsions), axis=1) # 长度为1000
S_synthetic_params = np.concatenate((S_synthetic_curvatures, S_synthetic_torsions), axis=1) # 长度为1000
C_srvf_synthetic_params = np.concatenate((C_srvf_synthetic_curvatures, C_srvf_synthetic_torsions), axis=1) # 长度为1000
U_srvf_synthetic_params = np.concatenate((U_srvf_synthetic_curvatures, U_srvf_synthetic_torsions), axis=1) # 长度为1000
V_srvf_synthetic_params = np.concatenate((V_srvf_synthetic_curvatures, V_srvf_synthetic_torsions), axis=1) # 长度为1000
S_srvf_synthetic_params = np.concatenate((S_srvf_synthetic_curvatures, S_srvf_synthetic_torsions), axis=1) # 长度为1000
C_params = np.concatenate((C_curvatures, C_torsions), axis=1)# 长度为87
U_params = np.concatenate((U_curvatures, U_torsions), axis=1)# 长度为87
V_params = np.concatenate((V_curvatures, V_torsions), axis=1)# 长度为87
S_params = np.concatenate((S_curvatures, S_torsions), axis=1)# 长度为87

merged_params = np.concatenate((C_synthetic_params, U_synthetic_params, V_synthetic_params, S_synthetic_params,
                                C_srvf_synthetic_params, U_srvf_synthetic_params, V_srvf_synthetic_params, S_srvf_synthetic_params,
                                C_params, U_params, V_params, S_params), axis=0)

color_labels = ['w'] * len(C_synthetic_params) + ['w'] * len(U_synthetic_params) + \
['w'] * len(V_synthetic_params) + ['w'] * len(S_synthetic_params) + \
['b'] * len(C_srvf_synthetic_params) + ['g'] *len(U_srvf_synthetic_params) + \
['r'] * len(V_srvf_synthetic_params) + ['orange'] * len(S_srvf_synthetic_params) +\
['b'] * len(C_params) + ['g'] * len(U_params) + ['r'] * len(V_params) + ['orange'] * len(S_params)

marker_labels = ['o'] * len(C_synthetic_params) + ['o'] * len(U_synthetic_params) +\
['o'] * len(V_synthetic_params) + ['o'] * len(S_synthetic_params) +\
['^'] * len(C_srvf_synthetic_params) + ['^'] * len(U_srvf_synthetic_params) +\
['^'] * len(V_srvf_synthetic_params) + ['^'] * len(S_srvf_synthetic_params) +\
['s'] * len(C_params) + ['s'] * len(U_params) + ['s'] * len(V_params) + ['s'] * len(S_params)


# 此时，color_labels 和 marker_labels 分别为颜色和标记的标签列表，与 merged_params 长度相等。

param_PCA = PCAHandler(merged_params, None, n_components=3, standardization=1)
param_PCA.PCA_training_and_test()

fig = plt.figure(dpi=100, figsize=(6, 5))
ax = fig.add_subplot(111)
# 迭代每一组数据并绘制散点图

for i in range(len(color_labels)):
    if i <8000:
        ax.scatter(param_PCA.train_res[i,0], param_PCA.train_res[i, 1], 
               c=color_labels[i], marker=marker_labels[i], alpha=0.7, edgecolors='white')
    else:
        ax.scatter(param_PCA.train_res[i,0], param_PCA.train_res[i, 1],
               c=color_labels[i], marker=marker_labels[i], alpha=0.7, edgecolors='black')    

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(geometry_dir+"PCA_total_with_params.png")
plt.close()
# plt.show()

end_time = datetime.now()
total_time = end_time - start_time
print(dir_formatted_time, "is done in", total_time.seconds, "seconds.")
log.close()
open_folder_in_explorer(bkup_dir)