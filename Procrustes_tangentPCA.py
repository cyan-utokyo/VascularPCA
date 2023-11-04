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
Procrustes_curves = np.array(parametrized_curves)

print (Procrustes_curves.shape)
i=30 # U
j=46 # S

log.write("- preprocessing_pca is not a SRVF PCA.\n")
preprocessing_pca = PCAHandler(Procrustes_curves.reshape(len(Procrustes_curves),-1), None, 20, PCA_STANDARDIZATION)
preprocessing_pca.PCA_training_and_test()
preprocess_curves = preprocessing_pca.inverse_transform_from_loadings(preprocessing_pca.train_res).reshape(len(preprocessing_pca.train_res), -1, 3)

Procrustes_curves = preprocess_curves
# SRVF计算
Procs_srvf_curves = np.zeros_like(Procrustes_curves)
for i in range(len(Procrustes_curves)):
    Procs_srvf_curves[i] = calculate_srvf((Procrustes_curves[i])/measure_length(Procrustes_curves[i]))
    print ("SRVF length:", measure_length(Procs_srvf_curves[i]))
    # print ("SRVF length by GPT4:", srvf_length(Procs_srvf_curves[i]))
log.write("SCALED to 1 before compute SRVF.\n")
log.write("according to A robust tangent PCA via shape restoration for shape variability analysis, tangent PCA is supposed to be conducted on SRVF.\n")

frechet_mean_srvf = compute_frechet_mean(Procs_srvf_curves)

#########################################
# 把srvf曲线做对数映射，得到切线空间的切向量
tangent_vectors = []
for curve in Procs_srvf_curves:
    # tangent_vector = discrete_curves_space.metric.log(point=curve, base_point=frechet_mean_curve)
    tangent_vector = discrete_curves_space.to_tangent(curve, frechet_mean_srvf)
    tangent_vectors.append(tangent_vector)
tangent_vectors = np.array(tangent_vectors)
# 把srvf曲线做对数映射，得到切线空间的切向量
#########################################

tpca = TangentPCA(metric=discrete_curves_space.metric, n_components=PCA_N_COMPONENTS)
log.write("discrete_curves_space.metric: <geomstats.geometry.discrete_curves.SRVMetric object\n")
# 拟合并变换数据到切线空间的主成分中
tpca.fit(tangent_vectors)
tangent_projected_data = tpca.transform(tangent_vectors)

principal_components = tpca.components_
# Assuming principal_components has the shape (n_components, n_sampling_points * n_dimensions)
principal_components_reshaped = principal_components.reshape((PCA_N_COMPONENTS, len(tangent_vectors[0]), 3))
# Now use exp on each reshaped component
curves_from_components = [
    discrete_curves_space.metric.exp(tangent_vec=component, base_point=frechet_mean_srvf)
    for component in principal_components_reshaped
]

reconstructed_curves = []
for idx in range(len(tangent_projected_data)):
    # This is your feature - a single point in PCA space representing the loadings for the first curve.
    feature = tangent_projected_data[idx]
    # Reconstruct the tangent vector from the feature.
    tangent_vector_reconstructed = sum(feature[i] * principal_components_reshaped[i] for i in range(len(feature)))
    # Map the tangent vector back to the curve space using the exponential map.
    reconstructed_srvf = discrete_curves_space.metric.exp(
        tangent_vec=tangent_vector_reconstructed, base_point=frechet_mean_srvf
    )
    reconstructed_curve = inverse_srvf(reconstructed_srvf, np.zeros(3))
    # print ("reconstructed_curve length:", measure_length(reconstructed_curve))# length=63
    
    reconstructed_curves.append(reconstructed_curve)
reconstructed_curves = np.array(reconstructed_curves)


def reconstruct_components(tpca, discrete_curves_space, frechet_mean_srvf, inverse_srvf_func):
    principal_components = tpca.components_
    # Assuming the shape of principal_components is (n_components, n_sampling_points * n_dimensions)
    principal_components_reshaped = principal_components.reshape(
        (tpca.n_components, -1, 3)
    )

    curves_from_components = []
    for component in principal_components_reshaped:
        # Map the tangent vector back to the curve space using the exponential map.
        srvf_curve = discrete_curves_space.metric.exp(
            tangent_vec=component, base_point=frechet_mean_srvf
        )
        # Apply the inverse SRVF to get the curve
        curve = inverse_srvf_func(srvf_curve, np.zeros(3))  # Assuming inverse_srvf function takes srvf_curve and a base point as input.
        curves_from_components.append(curve)
    curves_from_components = align_icp(curves_from_components, base_id=0)
    print ("First alignment done.")
    curves_from_components = align_procrustes(curves_from_components,base_id=0)
    # Visualize each reconstructed component curve
    return curves_from_components
# Usage
tangent_components = reconstruct_components(tpca, discrete_curves_space, frechet_mean_srvf, inverse_srvf)

sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('Blues'), norm=plt.Normalize(vmin=0, vmax=len(tangent_components)-1))
sm.set_array([])  # Only needed for the colorbar
fig1 = plt.figure(figsize=(13, 6),dpi=300)
ax1 = fig1.add_subplot(111)
for i in range(len(tangent_components)):
    color = sm.to_rgba(i)
    ax1.plot(tangent_components[i][:,0], label="component{}".format(i+1), color=color)
ax1.legend()
fig1.savefig(bkup_dir+"component.png")
plt.close(fig1)


###########################################################
# plot 各个主成分
def update_plot(angle):
    ax.view_init(elev=10., azim=angle)
fig, ax = plt.subplots(figsize=(6, 5), dpi=300, subplot_kw={'projection': '3d'})
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=plt.Normalize(vmin=0, vmax=len(tangent_components)-1))
sm.set_array([])  # Only needed for the colorbar

for i, component in enumerate(tangent_components):
    color = sm.to_rgba(i)
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

log.write("RECONSTRUCT_WITH_SRVF:"+str(0)+"\n")
Curvatures, Torsions = compute_synthetic_curvature_and_torsion(reconstructed_curves)

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
    # 初始化能量值列表
    energies = []
    for torsion, curvature in zip(np.array(selected_data_torsion), selected_data_curvature):
        energy = compute_geometry_param_energy(curvature, torsion)
        energies.append(energy)
    # 将计算的能量值存储在字典中
    param_dict[label]['Energy'] = energies

# 定义颜色映射
colors = {
    label: plt.cm.CMRmap((i+1)/(len(param_group_unique_labels)+1)) for i, label in enumerate(param_group_unique_labels)
}

# 创建一个图形和轴
fig1, ax1 = plt.subplots(dpi=300)
fig2, ax2 = plt.subplots(dpi=300)
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
for label in param_group_unique_labels:
    energies = param_dict[label]['Energy']
    curvatures, torsions = zip(*energies)
    total_curvature_energy.extend(curvatures)
    total_torsion_energy.extend(torsions)
    # 获取当前标签对应的大小值
    # sizes_for_label = y_prob_max[index : index + len(energies)]
    ax1.scatter(curvatures, torsions, 
               color=colors[label], 
               label=label, 
               alpha=0.9, 
               #s=sizes_for_label*sizes_for_label*75 
               ) 
    # for i in range(len(curvatures)):
    #     fontsize = 5
    #     ax1.annotate(Files[i].split("\\")[-1].split(".")[-2][:-7], (curvatures[i], torsions[i]), fontsize=fontsize)
    #     ax1.annotate(param_group[i], (curvatures[i], torsions[i]-0.0015), fontsize=fontsize, color=param_group_colors[param_group[i]])
    #     ax1.annotate(Typevalues[i], (curvatures[i], torsions[i]-0.0030), fontsize=fontsize, color=Typevalues_colors[Typevalues[i]] )
    # # 更新索引
    index += len(energies)
total_curvature_energy = np.array(total_curvature_energy)
total_torsion_energy = np.array(total_torsion_energy)
ax2.scatter(total_curvature_energy, total_torsion_energy, color="k", alpha=0.5, s=25)
# 计算线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(total_curvature_energy, total_torsion_energy)
# 计算预测值和置信区间
x_pred = np.linspace(total_curvature_energy.min(), total_curvature_energy.max(), 100)
y_pred = intercept + slope * x_pred
y_err = std_err * np.sqrt(1/len(total_curvature_energy) + (x_pred - np.mean(total_curvature_energy))**2 / np.sum((total_curvature_energy - np.mean(total_curvature_energy))**2))
conf_interval_upper = y_pred + 1.96 * y_err  # 95% 置信区间
conf_interval_lower = y_pred - 1.96 * y_err  # 95% 置信区间
ax1.plot(x_pred, y_pred, color="k", alpha=0.4)# , label='Fit: y = {:.2f} + {:.2f}x'.format(intercept, slope))
ax1.fill_between(x_pred, conf_interval_lower, conf_interval_upper, color='silver', alpha=0.15)# , label='95% CI')
# 比较合成形状
for ax in [ax1,ax2]:
    ax.set_xlabel('Curvature Energy')
    ax.set_ylabel('Torsion Energy')
    ax.set_title('Energy Scatter Plot by Label')
    ax.grid(linestyle='--', alpha=0.5)
ax1.legend()
fig1.savefig(bkup_dir+"Energy_Scatter_Plot_by_Label.png")
fig2.savefig(bkup_dir+"Energy_Scatter_Plot_by_Label2.png")
plt.close()
plt.close()

df = pd.DataFrame(tangent_projected_data, columns=[f'PC{i+1}' for i in range(PCA_N_COMPONENTS)])
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
plt.savefig(bkup_dir+"srvfPCA_total_Violinplot.png")
plt.close()

####################为SRVF PCA和geom param做sensitivity analysis####################
results = []
max_pcs_curvatures = {}
max_pcs_torsions = {}

# 为Curvatures和Torsion分别执行相同的操作
for variable_name, variable_data in [('Curvatures', Curvatures), ('Torsions', Torsions)]:
    # 遍历每个因变量
    for i in range(variable_data.shape[1]):
        y = variable_data[:, i].reshape(-1, 1)

        # 存储回归系数
        coefficients = {}

        # 遍历每个自变量
        for pc in range(PCA_N_COMPONENTS):
            X = tangent_projected_data[:, pc].reshape(-1, 1)

            model = LinearRegression().fit(X, y)
            coefficients[pc] = np.abs(model.coef_[0][0])*np.std(tangent_projected_data[:, pc] # 修正：该处之前是model.coef_[0][0]，乘上对应PC的标准差后得到的是大概的param的变化范围（被影响的程度），这个变动对所有landmark生效
                                                        )
        # 将coefficient_values分解为单独的列
        
        # 如果是Curvatures，使用绝对值来找出受影响最大的自变量
        if variable_name == 'Curvatures':
            max_pc = max(coefficients, key=lambda k: abs(coefficients[k]))
            max_coefficient = abs(coefficients[max_pc])
            coefficient_values = [abs(value) for value in coefficients.values()]
        else:  # 如果是Torsions，保持原有的计算方式
            max_pc = max(coefficients, key=lambda k: coefficients[k])
            max_coefficient = coefficients[max_pc]
            coefficient_values = list(coefficients.values())

        # 计算所有系数的均值和标准差
        mean_coefficient = np.mean(coefficient_values)
        std_coefficient = np.std(coefficient_values)
        coefficient_columns = {f'Coefficient_Value_{j+1}': coefficient_values[j] for j in range(len(coefficient_values))}
        # 将结果添加到列表中
        results.append({
            'Variable_Type': variable_name,
            'Dependent_Variable_Index': i,
            'Most_Influential_PCA_Component': max_pc,
            'Max_Coefficient': max_coefficient,
            'Mean_Coefficient': mean_coefficient,
            'Std_Coefficient': std_coefficient,
            **coefficient_columns  # 使用**来展开字典并将其合并到主字典中
        })

        # 根据变量类型存储每个因变量受影响最大的自变量编号
        if variable_name == 'Curvatures':
            max_pcs_curvatures[i] = max_pc
        elif variable_name == 'Torsions':
            max_pcs_torsions[i] = max_pc


# 将结果转换为DataFrame
results_df = pd.DataFrame(results)
# 从results_df中删除'coefficient_values'列
# results_df = results_df.drop(columns=['coefficient_values'])
# 将DataFrame输出为CSV文件
results_df.to_csv(bkup_dir+'regression_results.csv', index=False)



# print (results_df['Mean_Coefficient'].shape)
#print (np.std(all_srvf_pca.train_res,axis=0).shape) # 需要得到16个
# avg_curvatures = 16*results_df['Mean_Coefficient'][:61] # * np.mean(np.std(all_srvf_pca.train_res,axis=0))
# print ("avg_curvatures.shape:", avg_curvatures.shape)
# avg_torsions = 16*results_df['Mean_Coefficient'][61:] # * np.mean(np.std(all_srvf_pca.train_res,axis=0))
# print ("avg_torsions.shape:", avg_torsions.shape)

# 绘图
fig_x = 1
fig_shape = compute_frechet_mean(Procrustes_curves)[:, fig_x][3:]
# fig_shape = Procrustes_curves[7, :, fig_x][3:]
print("fig_shape.shape", fig_shape.shape)
colors = list(mcolors.TABLEAU_COLORS.keys())  # 获取一组颜色

fig = plt.figure(figsize=(13, 6),dpi=300)
# 定义GridSpec的行和列，然后设置行的高度比例。例如，这里我们设置第一个子图为3，第二个为1，所以第一个子图的高度是第二个的三倍。
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
# 使用GridSpec创建子图
ax = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax.plot(fig_shape, marker='o', linestyle='-', color="dimgray", label='Frechet Mean')
# ax.axvspan(0, 6, facecolor='dimgray', alpha=0.3)
# ax.axvspan(6, 25, facecolor='dimgray', alpha=0.2)
# ax.axvspan(25, 44, facecolor='dimgray', alpha=0.1)
# ax.axvspan(44, 54, facecolor='dimgray', alpha=0.2)
# ax.axvspan(54, 60, facecolor='dimgray', alpha=0.3)
ax.set_facecolor('whitesmoke')
# 添加barplot
indices = np.arange(len(fig_shape))
bar_width = 0.35
avg_curvatures = [np.mean(Curvatures[i]) for i in indices]
avg_torsions = [np.mean(np.abs(Torsions[i])) for i in indices]
ax2.bar(indices - bar_width/2, avg_curvatures, bar_width, label='Average Curvature', alpha=0.99, color="dimgray", edgecolor='k')
ax2.bar(indices + bar_width/2, avg_torsions, bar_width, label='Average Torsion', alpha=0.99, color="silver",edgecolor='k')
max_coeffs_curvatures = [entry['Max_Coefficient'] for entry in results if entry['Variable_Type'] == 'Curvatures']
max_coeffs_torsions = [entry['Max_Coefficient'] for entry in results if entry['Variable_Type'] == 'Torsions']
# 在已有的barplot上添加新的barplot
ax2.bar(indices - bar_width/2, max_coeffs_curvatures, bar_width, label='Max Coefficient Curvature', alpha=0.7,color="coral", edgecolor='k')
ax2.bar(indices + bar_width/2, np.abs(max_coeffs_torsions), bar_width,  label='Max Coefficient Torsion', alpha=0.7,color="royalblue",edgecolor='k')

# ax2.set_ylim(-0.2,0.8)
# ax.set_ylim(-15,7.5)
for i in range(len(fig_shape)):
    curv_pc = max_pcs_curvatures.get(i)
    tors_pc = max_pcs_torsions.get(i)
    if curv_pc is not None:
        ax.text(i, fig_shape[i] + 0.35, str(curv_pc+1), color=colors[curv_pc % len(colors)], ha='center')
    if tors_pc is not None:
        ax.text(i, fig_shape[i] - 0.35, str(tors_pc+1), color=colors[tors_pc % len(colors)], ha='center', va='top')

# plt.title('Frechet Mean with Influential PCA Components')
# ax.set_xlabel('Index')
ax.set_xticks([])
ax2.set_xlabel('Index')
ax.set_ylabel('Mean Shape')
ax2.set_ylabel('Geometry Parameter')
# ax2.set_xlim(0,75)
# ax2.legend()
plt.tight_layout()
plt.savefig(bkup_dir + "Frechet_Mean_with_Influential_PCA_Components.png")
plt.close()










end_time = datetime.now()
total_time = end_time - start_time
print(dir_formatted_time, "is done in", total_time.seconds, "seconds.")
log.close()
open_folder_in_explorer(bkup_dir)
