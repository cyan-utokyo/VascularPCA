import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA, KernelPCA
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.stats import zscore
import glob
from myvtk.GetMakeVtk import *
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
from dtw import *
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
from myvtk.MyRiemannCov import *
from scipy.signal import find_peaks
import geomstats.geometry.discrete_curves as dc
from geomstats.learning.pca import TangentPCA
from geomstats.geometry.discrete_curves import ElasticMetric, SRVMetric
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image
from myvtk.Myscores import *
warnings.filterwarnings("ignore")

# sns.set_context("talk", font_scale=0.8)
# sns.set_style("whitegrid")
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
# 创建离散曲线空间
r2 = Euclidean(dim=3)
srv_metric = SRVMetric(r2)
discrete_curves_space = DiscreteCurves(ambient_manifold=r2, k_sampling_points=64)

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
    # pt = pt - (pt[0]+[np.random.uniform(-0.0001,0.0001),np.random.uniform(-0.0001,0.0001),np.random.uniform(-0.0001,0.1)])
    # pt = pt/np.linalg.norm(pt[-1] - pt[0]) * 64
    unaligned_curves.append(pt[::-1])
    radii.append(Radius[::-1])
    pre_Curvatures.append(Curv[::-1])
    pre_Torsions.append(Tors[::-1])
unaligned_curves = np.array(unaligned_curves)
geometry_dir = mkdir(bkup_dir, "geometry")
Typevalues = np.array(Typevalues)

########################################

print ("全データ（{}）を読み込みました。".format(len(pre_files)))
print ("使用できるデータ：", len(Files))
for i in range(len(Files)):
    if "BG0014_R" in Files[i]:
        base_id = i
print ("base_id:{},casename:{}で方向調整する".format(base_id, Files[base_id]))

# 设定参考点 p，您可以根据需要修改这个点
p = np.array([0, 0, 1])  # 例如，令 p 在 z 轴上
unaligned_curves = np.array([align_endpoints(curve, p) for curve in unaligned_curves])

fig = plt.figure(figsize=(13, 6),dpi=300)
ax1 = fig.add_subplot(133)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(131)
for i in range(len(unaligned_curves)):
    ax1.plot(unaligned_curves[i][:,0], unaligned_curves[i][:,1])
    ax2.plot(unaligned_curves[i][:,0], unaligned_curves[i][:,2])
    ax3.plot(unaligned_curves[i][:,1], unaligned_curves[i][:,2])
fig.savefig(bkup_dir+"unaligned_curves.png")
plt.close(fig)


##################################################
#  从这里开始是对齐。                             #
#  To-Do: 需要保存Procrustes对齐后的              #
#  曲线各一条，作为后续的曲线对齐的基准。           #
##################################################

# a_curves = align_icp(unaligned_curves, base_id=base_id)
#print ("First alignment done.")
Procrustes_curves = align_procrustes(unaligned_curves, base_id=base_id)
print ("procrustes alignment done.")


# for i in range(len(Procrustes_curves)):
#     print ("length:", measure_length(Procrustes_curves[i]))
parametrized_curves = np.zeros_like(Procrustes_curves)
# aligned_curves = np.zeros_like(interpolated_curves)
for i in range(len(Procrustes_curves)):
    parametrized_curves[i] = arc_length_parametrize(Procrustes_curves[i])
Procrustes_curves = np.array(parametrized_curves)
# Procrustes_curves = np.array([align_endpoints(curve, p) for curve in Procrustes_curves])
# frechet_mean_shape=compute_frechet_mean(Procrustes_curves)

fig = plt.figure(figsize=(13, 6),dpi=300)
ax1 = fig.add_subplot(133)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(131)
for i in range(len(Procrustes_curves)):
    ax1.plot(Procrustes_curves[i][:,0], Procrustes_curves[i][:,1])
    ax2.plot(Procrustes_curves[i][:,0], Procrustes_curves[i][:,2])
    ax3.plot(Procrustes_curves[i][:,1], Procrustes_curves[i][:,2])
fig.savefig(bkup_dir+"Procrustes_curves.png")
plt.close(fig)






print (Procrustes_curves.shape)
i=30 # U
j=46 # S

log.write("- preprocessing_pca is not a SRVF PCA.\n")
preprocessing_pca = PCAHandler(Procrustes_curves.reshape(len(Procrustes_curves),-1), None, 20, PCA_STANDARDIZATION)
preprocessing_pca.PCA_training_and_test()
preprocess_curves = preprocessing_pca.inverse_transform_from_loadings(preprocessing_pca.train_res).reshape(len(preprocessing_pca.train_res), -1, 3)

Procrustes_curves = preprocess_curves

Procrustes_curves = np.array([align_endpoints(curve, p) for curve in Procrustes_curves])
# SRVF计算
Procs_srvf_curves = np.zeros_like(Procrustes_curves)
for i in range(len(Procrustes_curves)):
    # Procs_srvf_curves[i] = calculate_srvf((Procrustes_curves[i])/measure_length(Procrustes_curves[i]))
    Procs_srvf_curves[i] = calculate_srvf((Procrustes_curves[i])/np.linalg.norm(Procrustes_curves[i][-1] - Procrustes_curves[i][0]))
    # print ("SRVF length:", measure_length(Procs_srvf_curves[i]))
    # print ("SRVF length by GPT4:", srvf_length(Procs_srvf_curves[i]))
log.write("SCALED to (absolute length) 1 before compute SRVF.\n")
log.write("according to A robust tangent PCA via shape restoration for shape variability analysis, tangent PCA is supposed to be conducted on SRVF.\n")

TANGENT_BASE = "srvf_frechet"
#TANGENT_BASE = "BH0012_R"
if TANGENT_BASE == "frechet":
    frechet_mean_shape = compute_frechet_mean(Procrustes_curves)
    tangent_base = calculate_srvf(frechet_mean_shape)/measure_length(frechet_mean_shape)
    print ("using frechet as tangent_base.")
    log.write("using frechet as tangent_base.\n")
elif TANGENT_BASE == "srvf_frechet":
    tangent_base = compute_frechet_mean(Procs_srvf_curves)
    print ("using srvf_frechet as tangent_base.")
    log.write("using srvf_frechet as tangent_base.\n")
    frechet_mean_shape = compute_frechet_mean(Procrustes_curves)
    # print ("tangent_base_curve length:", measure_length(tangent_base_curve))
# else:
#     for i in range(len(Files)):
#         if TANGENT_BASE in Files[i]:
#             tangent_base_id = i
#     print ("using {} as tangent_base_id.".format(Files[tangent_base_id]))
#     log.write("using {} as tangent_base_id.\n".format(Files[tangent_base_id]))
#     tangent_base = Procs_srvf_curves[tangent_base_id]

#########################################
# 把srvf曲线做对数映射，得到切线空间的切向量
tangent_vectors = []
for curve in Procs_srvf_curves:
    tangent_vector = discrete_curves_space.to_tangent(curve, tangent_base)
    tangent_vectors.append(tangent_vector)
tangent_vectors = np.array(tangent_vectors)

# 把srvf曲线做对数映射，得到切线空间的切向量
#########################################


srvf_pca = PCAHandler(Procs_srvf_curves.reshape(len(Procs_srvf_curves),-1), None, PCA_N_COMPONENTS, PCA_STANDARDIZATION)
srvf_pca.PCA_training_and_test()

tpca = TangentPCA(metric=discrete_curves_space.metric, n_components=PCA_N_COMPONENTS)
# 拟合并变换数据到切线空间的主成分中
tpca.fit(tangent_vectors)
tangent_projected_data = tpca.transform(tangent_vectors)
dist_to_zero = []
for i in range(len(tangent_projected_data)):
    dist_to_zero.append(np.linalg.norm(tangent_projected_data[i]))

fig = plt.figure(figsize=(13, 6),dpi=300)
ax = fig.add_subplot(111)
ax.scatter(tangent_projected_data[:,0], tangent_projected_data[:,1], s=25, marker="o", color="k", alpha=0.5)
ax.scatter(srvf_pca.train_res[:,0], srvf_pca.train_res[:,1], s=25, marker="o", color="r", alpha=0.5)

# 选择一个颜色图
cmap_q = plt.cm.get_cmap('hsv', 87)  # 例如使用 HSV 颜色图

# 为每条线选择不同的颜色
for i in range(len(tangent_projected_data)):
    color = cmap_q(i)
    ax.plot([tangent_projected_data[i, 0], srvf_pca.train_res[i, 0]], 
            [tangent_projected_data[i, 1], srvf_pca.train_res[i, 1]], 
            color=color, alpha=0.5)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("tangent_VS_srvf_pca")
fig.savefig(bkup_dir+"tangent_projected_data.png")
plt.close(fig)


# fig = plt.figure(figsize=(13, 6),dpi=300)
# ax = fig.add_subplot(111)
# ax.scatter(tangent_projected_data[:,0], tangent_projected_data[:,1], s=25, marker="o", color="k", alpha=0.5)
# for i in range(len(tangent_projected_data)):
#     ax.annotate(Files[i].split('\\')[-1][:-12], (tangent_projected_data[i,0], tangent_projected_data[i,1]+0.05),color='r',
#                 fontsize=3)
#     ax.annotate(Typevalues[i], (tangent_projected_data[i,0], tangent_projected_data[i,1]),fontsize=10, color='b')   
# # plt.show()
# plt.savefig(bkup_dir+"annotation.png")
# plt.close(fig)


# def fit_kde(data, bandwidth=1):
#     kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
#     kde.fit(data)
#     return kde

# mean_tangent_projected_data = np.mean(tangent_projected_data, axis=0)
# std_tangent_projected_data = np.std(tangent_projected_data, axis=0)
# standardize_tangent_projected_data = zscore(tangent_projected_data)

def fit_gaussian(data):
    # 计算数据的均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # 创建一个高斯分布对象数组
    gaussians = [norm(loc=mean[i], scale=std[i]) for i in range(len(mean))]
    return gaussians

# 使用高斯方法拟合数据
tangent_pca_gaussian = fit_gaussian(tangent_projected_data)

# 从每个高斯分布中独立采样
sample_num = 1000  # 定义您想要的样本数
synthetic_features = np.array([g.rvs(sample_num) for g in tangent_pca_gaussian]).T


def from_tangentPCA_feature_to_curves(tpca, tangent_base, tangent_projected_data, inverse_srvf_func=inverse_srvf):
    principal_components = tpca.components_
    # Assuming principal_components has the shape (n_components, n_sampling_points * n_dimensions)
    principal_components_reshaped = principal_components.reshape((PCA_N_COMPONENTS, 64, 3)) # 64是采样点数
    # Now use exp on each reshaped component
    curves_from_components = [
        discrete_curves_space.metric.exp(tangent_vec=component, base_point=tangent_base)
        for component in principal_components_reshaped
    ]

    reconstructed_curves = []
    for idx in range(len(tangent_projected_data)):
        # This is your feature - a single point in PCA space representing the loadings for the first curve.
        feature = np.array(tangent_projected_data[idx])
        # Reconstruct the tangent vector from the feature.
        # print ("feature:", feature.shape)
        # print ("principal_components_reshaped:", principal_components_reshaped.shape)
        # print ("idx:", idx)
        tangent_vector_reconstructed = sum(feature[i] * principal_components_reshaped[i] for i in range(len(feature)))
        # Map the tangent vector back to the curve space using the exponential map.
        reconstructed_srvf = discrete_curves_space.metric.exp(
            tangent_vec=tangent_vector_reconstructed, base_point=tangent_base
        )
        reconstructed_curve = inverse_srvf(reconstructed_srvf, np.zeros(3))
        # print ("reconstructed_curve length:", measure_length(reconstructed_curve))# length=63
        
        reconstructed_curves.append(reconstructed_curve)
    reconstructed_curves = np.array(reconstructed_curves)
    return reconstructed_curves

reconstructed_curves = from_tangentPCA_feature_to_curves(tpca, tangent_base, tangent_projected_data, inverse_srvf_func=inverse_srvf)
reconstructed_synthetic_curves = from_tangentPCA_feature_to_curves(tpca, tangent_base, synthetic_features, inverse_srvf_func=inverse_srvf)
print ("done......")
def reconstruct_components(tpca, discrete_curves_space, tangent_base, inverse_srvf_func):
    principal_components = tpca.components_
    # Assuming the shape of principal_components is (n_components, n_sampling_points * n_dimensions)
    principal_components_reshaped = principal_components.reshape(
        (tpca.n_components, 64, 3)
    )

    curves_from_components = []
    for component in principal_components_reshaped:
        # Map the tangent vector back to the curve space using the exponential map.
        srvf_curve = discrete_curves_space.metric.exp(
            tangent_vec=component, base_point=tangent_base
        )
        # Apply the inverse SRVF to get the curve
        curve = inverse_srvf_func(srvf_curve, np.zeros(3))  # Assuming inverse_srvf function takes srvf_curve and a base point as input.
        curves_from_components.append(curve)
    #curves_from_components = align_icp(curves_from_components, base_id=0)
    #print ("First alignment done.")
    #curves_from_components = align_procrustes(curves_from_components,base_id=0)
    # Visualize each reconstructed component curve
    return curves_from_components
# Usage
tangent_components = reconstruct_components(tpca, discrete_curves_space, tangent_base, inverse_srvf)


# sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('tab20b'), norm=plt.Normalize(vmin=0, vmax=len(tangent_components)-1))
# sm.set_array([])  # Only needed for the colorbar
# fig1 = plt.figure(figsize=(13, 6),dpi=300)
# ax1 = fig1.add_subplot(111)
# for i in range(len(tangent_components)):
#     color = sm.to_rgba(i)
#     ax1.plot(tangent_components[i][:,0], label="component{}".format(i+1), color=color)
# ax1.legend()
# fig1.savefig(bkup_dir+"component.png")
# plt.close(fig1)


###########################################################
# plot 各个主成分
def update_plot(angle):
    ax.view_init(elev=10., azim=angle)
fig, ax = plt.subplots(figsize=(6, 5), dpi=300, subplot_kw={'projection': '3d'})
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('tab20b'), norm=plt.Normalize(vmin=0, vmax=len(tangent_components)-1))
sm.set_array([])  # Only needed for the colorbar

for i, component in enumerate(tangent_components):
    color = sm.to_rgba(i)
    component = align_endpoints(component, p)
    ax.plot(component[:, 0], component[:, 1], component[:, 2], color=color)

ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

# Create a colorbar
cbar = plt.colorbar(sm, ax=ax, pad=0.1, orientation='horizontal')
cbar.set_label('Principal Components')

angles = range(0, 360, 2)  # Angle range for the rotation
images = []

# Render each frame and append to images
for angle in angles:
    update_plot(angle)
    plt.draw()
    fig.canvas.draw()

    # Convert the figure to a PIL Image and append to list
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(Image.fromarray(image))

# Save to an animated GIF
images[0].save(
    bkup_dir + 'rotating_plot.gif',
    save_all=True,
    append_images=images[1:],
    duration=100,
    loop=0
)
plt.close(fig)
# plot 各个主成分
###########################################################
# Assuming tpca has an explained_variance_ratio_ attribute
explained_variance_ratio = tpca.explained_variance_ratio_
# Calculate the cumulative variance
cumulative_variance = np.cumsum(explained_variance_ratio)
# Plot the CCR curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
plt.title('Cumulative Captured Ratio (CCR) of PCA')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.savefig(bkup_dir+"CCR.png")
plt.close()
###########################################################

Curvatures, Torsions = compute_synthetic_curvature_and_torsion(reconstructed_curves)
synthetic_Curvatures, synthetic_Torsions = compute_synthetic_curvature_and_torsion(reconstructed_synthetic_curves)
post_reconstructed_synthetic_curves = []
post_synthetic_Curvatures = []
post_synthetic_Torsions = []
post_synthetic_features = []
count = 0
for i in range(len(reconstructed_synthetic_curves)):
    Curvature_energy, Torsion_energy = compute_geometry_param_energy(synthetic_Curvatures[i], synthetic_Torsions[i])
    Tortuosity = compute_tortuosity(reconstructed_synthetic_curves[i])
    if 0.01<Curvature_energy < 0.08 and Torsion_energy < 0.22 and 1.6< Tortuosity < 2.7:
        post_reconstructed_synthetic_curves.append(reconstructed_synthetic_curves[i])
        post_synthetic_Curvatures.append(synthetic_Curvatures[i])
        post_synthetic_Torsions.append(synthetic_Torsions[i])
        post_synthetic_features.append(synthetic_features[i])
        count += 1
    if count == 320:
        break
reconstructed_synthetic_curves = np.array(post_reconstructed_synthetic_curves)
synthetic_Curvatures = np.array(post_synthetic_Curvatures)
synthetic_Torsions = np.array(post_synthetic_Torsions)
synthetic_features = np.array(post_synthetic_features)


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


synthetic_torsion_param_group = []
synthetic_param_group = []
synthetic_HT_group = []
synthetic_LT_group = []
for i in range(len(synthetic_Torsions)):
    if np.mean(np.abs(synthetic_Torsions[i])) > average_of_means_torsions and np.std(synthetic_Torsions[i]) > average_of_std_torsions:
        synthetic_torsion_param_group.append("HMHS")
        synthetic_param_group.append("HT")
        synthetic_HT_group.append(i)
    elif np.mean(np.abs(synthetic_Torsions[i])) > average_of_means_torsions and np.std(synthetic_Torsions[i]) <= average_of_std_torsions:
        synthetic_torsion_param_group.append("HMLS")
        synthetic_param_group.append("HT")
        synthetic_HT_group.append(i)
    elif np.mean(np.abs(synthetic_Torsions[i])) <= average_of_means_torsions and np.std(synthetic_Torsions[i]) > average_of_std_torsions:
        synthetic_torsion_param_group.append("LMHS")
        synthetic_param_group.append("LT")
        synthetic_LT_group.append(i)
    elif np.mean(np.abs(synthetic_Torsions[i])) <= average_of_means_torsions and np.std(synthetic_Torsions[i]) <= average_of_std_torsions:
        synthetic_torsion_param_group.append("LMLS")
        synthetic_param_group.append("LT")
        synthetic_LT_group.append(i)


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

synthetic_LT_curvatures = synthetic_Curvatures[synthetic_LT_group]
synthetic_HT_curvatures = synthetic_Curvatures[synthetic_HT_group]
print ("len(synthetic_LT_curvatures):", len(synthetic_LT_curvatures))
print ("len(synthetic_HT_curvatures):", len(synthetic_HT_curvatures))
synthetic_quad_param_group = []
for i in range(len(synthetic_Curvatures)):
    synthetic_curvature_mean = np.mean(synthetic_Curvatures[i])
    if synthetic_param_group[i] == 'LT':
        threshold = average_of_LT_curvature
    elif synthetic_param_group[i] == 'HT':
        threshold = average_of_HT_curvature
    if synthetic_curvature_mean > threshold:
        synthetic_quad_param_group.append(synthetic_param_group[i] + 'HC')
        synthetic_param_group[i] = synthetic_torsion_param_group[i] + 'HC'
    else:
        synthetic_quad_param_group.append(synthetic_param_group[i] + 'LC')
        synthetic_param_group[i] = synthetic_torsion_param_group[i] + 'LC'


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
synthetic_param_dict = {label: {} for label in param_group_unique_labels}
# 填充字典
for label in param_group_unique_labels:
    # 使用布尔索引来选择与当前标签对应的数据
    selected_data_torsion = [Torsions[i] for i, tag in enumerate(quad_param_group) if tag == label]
    selected_data_curvature = [Curvatures[i] for i, tag in enumerate(quad_param_group) if tag == label]
    synthetic_selected_data_torsion = [synthetic_Torsions[i] for i, tag in enumerate(synthetic_quad_param_group) if tag == label]
    synthetic_selected_data_curvature = [synthetic_Curvatures[i] for i, tag in enumerate(synthetic_quad_param_group) if tag == label]
    # 将选择的数据转换为numpy array并保存到字典中
    param_dict[label]['Torsion'] = np.array(selected_data_torsion)
    param_dict[label]['Curvature'] = np.array(selected_data_curvature)
    synthetic_param_dict[label]['Torsion'] = np.array(synthetic_selected_data_torsion)
    synthetic_param_dict[label]['Curvature'] = np.array(synthetic_selected_data_curvature)
    # 初始化能量值列表
    energies = []
    for torsion, curvature in zip(np.array(selected_data_torsion), selected_data_curvature):
        energy = compute_geometry_param_energy(curvature, torsion)
        energies.append(energy)
    synthetic_energies = []
    for torsion, curvature in zip(np.array(synthetic_selected_data_torsion), synthetic_selected_data_curvature):
        energy = compute_geometry_param_energy(curvature, torsion)
        synthetic_energies.append(energy)
    # 将计算的能量值存储在字典中
    param_dict[label]['Energy'] = energies
    synthetic_param_dict[label]['Energy'] = synthetic_energies

    selected_data_curve = [Procrustes_curves[i] for i, tag in enumerate(quad_param_group) if tag == label]
    synthetic_selected_data_curve = [reconstructed_synthetic_curves[i] for i, tag in enumerate(synthetic_quad_param_group) if tag == label]
    tortuosity = []
    synthetic_tortuosity = []
    for curve in selected_data_curve:
        tortuosity.append(compute_tortuosity(curve))
    for curve in synthetic_selected_data_curve:
        synthetic_tortuosity.append(compute_tortuosity(curve))
    param_dict[label]['Tortuosity'] = np.array(tortuosity)
    synthetic_param_dict[label]['Tortuosity'] = np.array(synthetic_tortuosity)

# 定义颜色映射
colors = {
    label: plt.cm.CMRmap((i+1)/(len(param_group_unique_labels)+1)) for i, label in enumerate(param_group_unique_labels)
}

# Calculate centroids and scores for each label
energy_centroids = {}
energy_scores = {label: [] for label in param_group_unique_labels}
# Calculate the centroid of energies for each label
for label in param_group_unique_labels:
    energy_centroids[label] = compute_energy_centroid(param_dict[label]['Energy'])

# Calculate the scores for each energy value based on its distance from the centroid
fig = plt.figure(figsize=(8, 5), dpi=300)
ax = fig.add_subplot(111)
for i in range(len(Procrustes_curves)):
    energy = compute_geometry_param_energy(Curvatures[i], Torsions[i])
    ax.scatter(energy[0], energy[1], color=colors[quad_param_group[i]], alpha=0.9, s=25, marker="${}$".format(Typevalues[i]))
    # ax.annotate(Files[i].split("\\")[-1].split(".")[-2][:-7], (energy[0], energy[1]), fontsize=5)
    print (quad_param_group[i], Typevalues[i])
    shape_score = {}
    for label in param_group_unique_labels:
        energy_score = score_energy(energy, energy_centroids[label])
        shape_score[label] = energy_score
    print (shape_score)
    print ("-"*20)
plt.savefig(bkup_dir+"centroid_classification.png")
plt.close()

##########################################################
# 计算距离矩阵


# 创建一个图形和轴
fig1, ax1 = plt.subplots(dpi=300)
fig2, ax2 = plt.subplots(dpi=300)
fig3 = plt.figure(figsize=(8, 5), dpi=300)
ax3 = fig3.add_subplot(111, projection='3d')
# 初始化一个索引来跟踪y_prob_max中的当前位置
index = 0
# 绘制散点图
# 定义颜色映射
Typevalues_colors = {
    label: plt.cm.gnuplot((i)/4) for i, label in enumerate(set(Typevalues))
}
param_group_colors = {
    label: plt.cm.gnuplot((i)/8) for i, label in enumerate(set(param_group))
}
total_curvature_energy = []
total_torsion_energy = []
synthetic_total_curvature_energy = []
synthetic_total_torsion_energy = []
total_tortuosity = []
synthetic_total_tortuosity = []
for label in param_group_unique_labels:
    energies = param_dict[label]['Energy']
    synthetic_energies = synthetic_param_dict[label]['Energy']
    tortuosity = param_dict[label]['Tortuosity']
    total_tortuosity.extend(tortuosity)
    synthetic_tortuosity = synthetic_param_dict[label]['Tortuosity']
    synthetic_total_tortuosity.extend(synthetic_tortuosity)
    curvatures, torsions = zip(*energies)
    synthetic_curvatures, synthetic_torsions = zip(*synthetic_energies)
    total_curvature_energy.extend(curvatures)
    total_torsion_energy.extend(torsions)
    synthetic_total_curvature_energy.extend(synthetic_curvatures)
    synthetic_total_torsion_energy.extend(synthetic_torsions)
    # 获取当前标签对应的大小值
    # sizes_for_label = y_prob_max[index : index + len(energies)]
    ax1.scatter(curvatures, torsions, 
               color=colors[label], 
               label=label, 
               alpha=0.5, 
               s = tortuosity*tortuosity*20) 
    fontsize = 5
    for i in range(len(curvatures)):
        ax1.annotate(round(tortuosity[i],2), (curvatures[i], torsions[i]), fontsize=fontsize)
    ax2.scatter(synthetic_curvatures, synthetic_torsions,
                color=colors[label],
                label=label,
                alpha=0.5,
                s = synthetic_tortuosity*synthetic_tortuosity*20)
    # for i in range(len(curvatures)):
    #     fontsize = 5
    #     ax1.annotate(Files[i].split("\\")[-1].split(".")[-2][:-7], (curvatures[i], torsions[i]), fontsize=fontsize)
    #     ax1.annotate(param_group[i], (curvatures[i], torsions[i]-0.0015), fontsize=fontsize, color=param_group_colors[param_group[i]])
    #     ax1.annotate(Typevalues[i], (curvatures[i], torsions[i]-0.0030), fontsize=fontsize, color=Typevalues_colors[Typevalues[i]] )
    # # 更新索引
    ax3.scatter3D(synthetic_curvatures,synthetic_torsions,synthetic_tortuosity,
                color=colors[label],
                label=label,
                alpha=0.3,
                marker='^')
    ax3.scatter3D(curvatures,torsions,tortuosity,
                color=colors[label],
                label=label,
                alpha=0.9,
                marker='o')
    index += len(energies)

total_curvature_energy = np.array(total_curvature_energy)
total_torsion_energy = np.array(total_torsion_energy)
synthetic_total_curvature_energy = np.array(synthetic_total_curvature_energy)
synthetic_total_torsion_energy = np.array(synthetic_total_torsion_energy)
total_tortuosity = np.array(total_tortuosity)
synthetic_total_tortuosity = np.array(synthetic_total_tortuosity)
# ax2.scatter(total_curvature_energy, total_torsion_energy, color="k", alpha=0.5, s=25)
# 计算线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(total_curvature_energy, total_torsion_energy)
synthetic_slope, synthetic_intercept, synthetic_r_value, synthetic_p_value, synthetic_std_err = stats.linregress(synthetic_total_curvature_energy, synthetic_total_torsion_energy)
# 计算预测值和置信区间
x_pred = np.linspace(total_curvature_energy.min(), total_curvature_energy.max(), 100)
synthetic_x_pred = np.linspace(synthetic_total_curvature_energy.min(), synthetic_total_curvature_energy.max(), 100)
y_pred = intercept + slope * x_pred
synthetic_y_pred = synthetic_intercept + synthetic_slope * synthetic_x_pred
y_err = std_err * np.sqrt(1/len(total_curvature_energy) + (x_pred - np.mean(total_curvature_energy))**2 / np.sum((total_curvature_energy - np.mean(total_curvature_energy))**2))
conf_interval_upper = y_pred + 1.96 * y_err  # 95% 置信区间
conf_interval_lower = y_pred - 1.96 * y_err  # 95% 置信区间
ax1.plot(x_pred, y_pred, color="k", alpha=0.4, label='Fit: y = {:.2f} + {:.2f}x'.format(intercept, slope))
ax1.fill_between(x_pred, conf_interval_lower, conf_interval_upper, color='silver', alpha=0.15)# , label='95% CI')
# 比较合成形状
ax2.plot(x_pred, y_pred, color="k", alpha=0.4, linestyle=":",label='Fit: y = {:.2f} + {:.2f}x'.format(intercept, slope))
ax2.plot(synthetic_x_pred, synthetic_y_pred, color="k", alpha=0.4,label='Fit: y = {:.2f} + {:.2f}x'.format(synthetic_intercept, synthetic_slope))
for ax in [ax1,ax2]:
    ax.set_xlabel('Curvature Energy')
    ax.set_ylabel('Torsion Energy')
    ax.set_title('Energy Scatter Plot by Label')
    ax.set_ylim(-0.2, 0.5)
    ax.grid(linestyle='--', alpha=0.5)
ax1.legend()
ax2.legend()
ax3.legend()
ax3.set_xlabel('Curvature')
ax3.set_ylabel('Torsion')
ax3.set_zlabel('Tortuosity')
# fig3.show()
fig1.savefig(bkup_dir+"Energy_Scatter_Plot_by_Label.png")
fig2.savefig(bkup_dir+"Energy_Scatter_Plot_by_Label=synthetic.png")
fig3.savefig(bkup_dir+"Tortuosity_Scatter_Plot_by_Label.png")
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)

df = pd.DataFrame(tangent_projected_data, columns=[f'PC{i+1}' for i in range(PCA_N_COMPONENTS)])
df['Type'] = quad_param_group
# 创建一个4x4的子图网格
fig, axes = plt.subplots((PCA_N_COMPONENTS//Multi_plot_rows), Multi_plot_rows, figsize=(16, 20))
# 为每个主成分绘制violinplot
for i in range(PCA_N_COMPONENTS):
    ax = axes[i // Multi_plot_rows, i % Multi_plot_rows]
    # sns.violinplot(x='Type', y=f'PC{i+1}', data=df, ax=ax, inner='quartile', palette=colors)  # inner='quartile' 在violin内部显示四分位数
    sns.boxplot(x='Type', y=f'PC{i+1}', data=df, ax=ax, palette=colors, width=0.25)  # inner='quartile' 在violin内部显示四分位数
    ax.set_title(f'Principal Component {i+1}')
    ax.set_ylabel('')  # 移除y轴标签，使得图更加简洁
plt.tight_layout()
plt.savefig(bkup_dir+"tangentPCA_total_boxplot.png")
plt.close()


####################BraVa Data的各PC####################
def map_quad_params_to_int(quad_params):
    unique_params, inverse_indices = np.unique(quad_params, return_inverse=True)
    # 打印每个 unique param 对应的 quad_param_group 中的元素
    mapping = {unique_param: quad_param for unique_param, quad_param in zip(unique_params, range(len(unique_params)))}
    return inverse_indices, mapping
# 假设 quad_param_group 是一个已经定义的数组
quad_param_group_mapped, mapping = map_quad_params_to_int(quad_param_group)
# 打印映射结果
print("Mapped Indices: ", quad_param_group_mapped)
print("Mapping: ", mapping)
vtk_dir = mkdir(bkup_dir , "vtk")

for i in range(PCA_N_COMPONENTS):
    single_component_feature = np.zeros_like(tangent_projected_data)
    single_component_feature[:,i] = tangent_projected_data[:,i]
    single_reconstructed_curves = []
    fig = plt.figure(figsize=(10,2),dpi=300)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig3 = plt.figure(figsize=(8, 5), dpi=300)
    ax31 = fig3.add_subplot(111)
    single_reconstructed_curves = from_tangentPCA_feature_to_curves(tpca, tangent_base, single_component_feature, inverse_srvf_func=inverse_srvf)
    single_reconstructed_curvature, single_reconstructed_torsion = compute_synthetic_curvature_and_torsion(single_reconstructed_curves)
    # print ("single_reconstructed_curvature.shape:", single_reconstructed_curvature.shape)
    # print ("Curvautres.shape:", Curvatures.shape)
    curvature_mean_values = np.mean(single_reconstructed_curvature, axis=0)
    curvature_std_dev = np.std(single_reconstructed_curvature, axis=0)
    torsion_mean_values = np.mean(single_reconstructed_torsion, axis=0)
    torsion_std_dev = np.std(single_reconstructed_torsion, axis=0)
    ax1.plot(curvature_mean_values, label=r'$\bar{\kappa}$')
    ax2.plot(torsion_mean_values, label=r'$\bar{\tau}$')
    ax1.errorbar(range(len(curvature_mean_values)), curvature_mean_values, yerr=curvature_std_dev, fmt='-o')
    ax2.errorbar(range(len(torsion_mean_values)), torsion_mean_values, yerr=torsion_std_dev, fmt='-o')
    all_single_curvatures = [curvature for curve in single_reconstructed_curvature for curvature in curve]
    all_single_torsions = [torsion for curve in single_reconstructed_torsion for torsion in curve]
    all_curvature = [curvature for curve in Curvatures for curvature in curve]
    all_torsion = [torsion for curve in Torsions for torsion in curve]
    vtk_file_name = vtk_dir + f"PC{i+1}.vtk"
    write_vtk_line(vtk_file_name, 
                   single_reconstructed_curves, 
                   quad_param_group_mapped, 
                   single_component_feature[:,i], 
                   all_single_curvatures, all_single_torsions, all_curvature, all_torsion)
    for label,key in zip(mapping.values(), mapping.keys()):
    # 筛选出特定标签的数据
        data = single_component_feature[:, i][np.array(quad_param_group_mapped) == label]
        mean_label = np.mean(data)
        std_label = np.std(data)
        # 创建用于绘制高斯分布的数据点
        x_values = np.linspace(min(data), max(data), 1000)
        y_values = norm.pdf(x_values, loc=mean_label, scale=std_label)
        # 绘制高斯分布
        ax31.plot(x_values, y_values, label=f'{key} Gaussian', color=colors[key])

    ax31.legend()
    ax1.set_title(f'PC{i+1}')
    ax2.set_title(f'PC{i+1}')
    ax1.set_xlabel('Sampling Points')
    ax2.set_xlabel('Sampling Points')
    ax1.set_ylabel('Curvature')
    ax2.set_ylabel('Torsion')
    fig.savefig(bkup_dir+f"PC{i+1}.png")
    fig3.savefig(bkup_dir+f"PC{i+1}_gaussian.png")
    plt.close(fig3)
    plt.tight_layout()
    plt.close(fig)
    # plt.show()

reconstructed_curves = np.array([align_endpoints(curve, p) for curve in reconstructed_curves])



fig1 = plt.figure(figsize=(12, 3),dpi=300)
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(figsize=(12, 3),dpi=300)
ax2 = fig2.add_subplot(111)
frechet_mean_shape = align_endpoints(frechet_mean_shape, p)
# ax.plot(frechet_mean_shape[:,0], label="Mean shape", color="k")  
curvature_means = {key: [] for key in mapping.keys()}
torsion_means = {key: [] for key in mapping.keys()}
for label,key in zip(mapping.values(), mapping.keys()):
    data = reconstructed_curves[np.array(quad_param_group_mapped) == label]
    curvature = Curvatures[np.array(quad_param_group_mapped) == label]
    torsion = Torsions[np.array(quad_param_group_mapped) == label]
    # for curve in data:
    #     ax.scatter(range(len(curve)), curve[:,0], color=colors[key], alpha=0.5, marker="+", s=10)
    print ("colorkey", colors[key])
    # 初始化存储均值的列表
    curvature_means_per_loci = []
    torsion_means_per_loci = []
    for loci in range(len(curvature[0])):
        curvature_means[key].append(np.mean(curvature[:, loci]))
        torsion_means[key].append(np.mean(torsion[:, loci]))
        box = ax1.boxplot(curvature[:, loci], positions=[loci], widths=0.3, 
                        showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor=colors[key], color=colors[key]))
        box = ax2.boxplot(torsion[:, loci], positions=[loci], widths=0.3, 
                        showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor=colors[key], color=colors[key]))
ax1.set_xlabel('Sampling Points')
ax1.set_ylabel('Curvature')
ax1.set_xticklabels(ax1.get_xticks(), rotation=45)
ax2.set_xlabel('Sampling Points')
ax2.set_ylabel('Torsion')
ax2.set_xticklabels(ax2.get_xticks(), rotation=45)
plt.tight_layout()
fig1.savefig(bkup_dir+"X_axis_of_curvature.png")
plt.close(fig1)
fig2.savefig(bkup_dir+"X_axis_of_torsion.png")
plt.close(fig2)

# 初始化存储均值的列表
curvature_means_per_loci = [[] for _ in range(len(curvature[0]))]
torsion_means_per_loci = [[] for _ in range(len(torsion[0]))]

# 聚合不同类别的均值
for loci in range(len(curvature[0])):
    for key in curvature_means:
        curvature_means_per_loci[loci].append(curvature_means[key][loci])
        torsion_means_per_loci[loci].append(torsion_means[key][loci])

# 计算类别均值的标准差
curvature_std_devs = [np.std(means) for means in curvature_means_per_loci]
torsion_std_devs = [np.std(means) for means in torsion_means_per_loci]

# 标准差降序排序并获取前10个最大的loci
# top10_diff_curvature_loci = np.argsort(curvature_std_devs)[::-1][:10]
# top10_diff_torsion_loci = np.argsort(torsion_std_devs)[::-1][:10]

diff_curvature_loci = np.argsort(curvature_std_devs)
diff_torsion_loci = np.argsort(torsion_std_devs)

# print(f"Loci with the largest differences in curvature means: {top10_diff_curvature_loci}")
# print(f"Loci with the largest differences in torsion means: {top10_diff_torsion_loci}")


fig1 = plt.figure(figsize=(3,12),dpi=300)
ax1 = fig1.add_subplot(111)
ax1.plot(frechet_mean_shape[:,0], frechet_mean_shape[:,2], label="Mean shape", color="dimgray")  
fig2 = plt.figure(figsize=(3,12),dpi=300)
ax2 = fig2.add_subplot(111)
ax2.plot(frechet_mean_shape[:,0], frechet_mean_shape[:,2], label="Mean shape", color="dimgray")  
fig3 = plt.figure(figsize=(3,12),dpi=300)
ax3 = fig3.add_subplot(111)
ax3.plot(frechet_mean_shape[:,0], frechet_mean_shape[:,2], label="Mean shape", color="dimgray")

for label,key in zip(mapping.values(), mapping.keys()):
    data = reconstructed_curves[np.array(quad_param_group_mapped) == label]
    for curve in data:
        for loci in range(len(curve)-5):
            ax1.scatter(curve[loci,0], curve[loci,2], color=colors[key], 
                        alpha=np.exp(-15 * np.where(diff_curvature_loci == loci)[0]/len(diff_curvature_loci)), marker="x", s=20)
            ax2.scatter(curve[loci,0], curve[loci,2], color=colors[key], 
                        alpha=np.exp(-15 * np.where(diff_torsion_loci == loci)[0]/len(diff_torsion_loci)), marker="+", s=20)
            ax3.scatter(curve[loci,0], curve[loci,2], color=colors[key], 
                        alpha=np.exp(-15 * np.where(diff_curvature_loci == loci)[0]/len(diff_curvature_loci)), marker="x", s=20)
            ax3.scatter(curve[loci,0], curve[loci,2], color=colors[key], 
                        alpha=np.exp(-15 * np.where(diff_torsion_loci == loci)[0]/len(diff_torsion_loci)), marker="+", s=20)
for loci in range(2, len(curve)-5):
    ax1.scatter(frechet_mean_shape[loci,0], frechet_mean_shape[loci,2], marker='o',color="dimgray",
                alpha=np.exp(-15 * np.where(diff_curvature_loci == loci)[0]/len(diff_curvature_loci)))
    ax2.scatter(frechet_mean_shape[loci,0], frechet_mean_shape[loci,2], marker='o',color="dimgray",
                alpha=np.exp(-15 * np.where(diff_torsion_loci == loci)[0]/len(diff_torsion_loci)))
    ax3.scatter(frechet_mean_shape[loci,0], frechet_mean_shape[loci,2], marker='o',color="dimgray",
                alpha=np.exp(-15 * np.where(diff_curvature_loci == loci)[0]/len(diff_curvature_loci)))
    ax3.scatter(frechet_mean_shape[loci,0], frechet_mean_shape[loci,2], marker='o',color="dimgray",
                alpha=np.exp(-15 * np.where(diff_torsion_loci == loci)[0]/len(diff_torsion_loci)))

# ax1.set_xlabel('Sampling Points')
# ax1.set_ylabel('X axis')
ax1.set_xticklabels(ax1.get_xticks(), rotation=45)
plt.tight_layout()
fig1.savefig(bkup_dir+"X_axis_of_curvature_.png")
plt.close(fig1)
# ax2.set_xlabel('Sampling Points')
# ax2.set_ylabel('X axis')
ax2.set_xticklabels(ax2.get_xticks(), rotation=45)
plt.tight_layout()
fig2.savefig(bkup_dir+"X_axis_of_torsion_.png")
plt.close(fig2)
# ax3.set_xlabel('Sampling Points')
# ax3.set_ylabel('X axis')
ax3.set_xticklabels(ax3.get_xticks(), rotation=45)
plt.tight_layout()
fig3.savefig(bkup_dir+"X_axis_of_curvature_and_torsion_.png")





fig = plt.figure(figsize=(8, 4),dpi=300)
ax = fig.add_subplot(111)
for i in range(PCA_N_COMPONENTS):
    ax.boxplot(tangent_projected_data[:,i], positions=[i], widths=0.6, showfliers=False)
ax.set_xticks(range(PCA_N_COMPONENTS),angle=45)
ax.set_xticklabels([f'PC{i+1}' for i in range(PCA_N_COMPONENTS)])
ax.set_xlabel('Principal Components')
ax.set_ylabel('Loadings')
ax.set_title('Loadings of Principal Components')
plt.tight_layout()
plt.savefig(bkup_dir+"Loadings_of_Principal_Components.png")
plt.close(fig)



fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
fig2 = plt.figure(dpi=300)
ax2 = fig2.add_subplot(111)
for label in param_group_unique_labels:
    print (label)
    print ("real data:", param_dict[label]['Torsion'].shape)
    print ("synthetic data:", synthetic_param_dict[label]['Torsion'].shape)
    ax.scatter(synthetic_features[np.array(synthetic_quad_param_group) == label][:,0], 
               synthetic_features[np.array(synthetic_quad_param_group) == label][:,1],
               color=colors[label], 
               label=label, 
               alpha=0.5,
               s=synthetic_param_dict[label]['Tortuosity']*synthetic_param_dict[label]['Tortuosity']*20)
    # ax.scatter(tangent_projected_data[np.array(quad_param_group) == label][:,0], 
    #            tangent_projected_data[np.array(quad_param_group) == label][:,1],
    #            color=colors[label], 
    #            label=label+ " real",
    #            alpha=0.7, 
    #            marker="x",
    #            s=50) 
    ax2.scatter(tangent_projected_data[np.array(quad_param_group) == label][:,0], 
               tangent_projected_data[np.array(quad_param_group) == label][:,1],
               color=colors[label], 
               label=label, 
               alpha=0.7,)
    for i in range(10):
        makeVtkFile(vtk_dir + label + "_synthetic_" + str(i) + ".vtk",reconstructed_synthetic_curves[np.array(synthetic_quad_param_group) == label][i], [],[])
        makeVtkFile(vtk_dir + label + "_real_" + str(i) + ".vtk",reconstructed_curves[np.array(quad_param_group) == label][i],[],[])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax.legend()
ax2.legend()
fig.savefig(bkup_dir + "pcaloadings_plot_both.png")
fig2.savefig(bkup_dir + "pcaloadings_plot_real.png")
plt.close(fig1)
plt.close(fig2)







end_time = datetime.now()
total_time = end_time - start_time
print(dir_formatted_time, "is done in", total_time.seconds, "seconds.")
log.close()
open_folder_in_explorer(bkup_dir)
