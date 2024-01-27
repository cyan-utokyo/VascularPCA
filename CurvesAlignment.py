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
from geomstats.learning.pca import TangentPCA
from geomstats.geometry.discrete_curves import ElasticMetric, SRVMetric
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image
from myvtk.Myscores import *
from myvtk.MytangentPCA import *
warnings.filterwarnings("ignore")
import matplotlib as mpl
import dill as pickle
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from myvtk.MyAlignment import *

# mpl.rcParams['xtick.labelsize'] = 18
# mpl.rcParams['ytick.labelsize'] = 18
# mpl.rcParams['axes.labelsize'] = 18
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
    # pt = interpolate_pt(pt, 640)
    # pt = pt - (pt[0]+[np.random.uniform(-0.0001,0.0001),np.random.uniform(-0.0001,0.0001),np.random.uniform(-0.0001,0.1)])
    # pt = pt/np.linalg.norm(pt[-1] - pt[0]) * 64
    unaligned_curves.append(pt[::-1])
    radii.append(Radius[::-1])
    pre_Curvatures.append(Curv[::-1])
    pre_Torsions.append(Tors[::-1])
unaligned_curves = np.array(unaligned_curves)
# geometry_dir = mkdir(bkup_dir, "geometry")
Typevalues = np.array(Typevalues)

POINTS_NUM = pt.shape[0]
print ("POINTS_NUM:", POINTS_NUM)

r3 = Euclidean(dim=3)
srv_metric = SRVMetric(r3)
discrete_curves_space = DiscreteCurves(ambient_manifold=r3, k_sampling_points=POINTS_NUM)
print (discrete_curves_space.metric)
########################################

log.write("Data used: {}".format(len(Files)))
for i in range(len(Files)):
    if "BG0014_R" in Files[i]:
        base_id = i
log.write("Alignment reference data (base_id):{},casename:{}".format(base_id, Files[base_id]))
# print (siphon_idx)

##################################################
#  从这里开始是对齐。                             #
#  To-Do: 需要保存Procrustes对齐后的              #
#  曲线各一条，作为后续的曲线对齐的基准。           #
##################################################

Procrustes_curves = copy.deepcopy(unaligned_curves)
temp_mean_shapes = []
interpolate_pt_num = 640
# fig = plt.figure(figsize=(13, 6),dpi=300)
# ax1 = fig.add_subplot(111)
interpolated_procrustes_curves = []

for i in range(len(Procrustes_curves)):
    interpolated_procrustes_curves.append(interpolate_pt(Procrustes_curves[i], interpolate_pt_num))
interpolated_procrustes_curves = np.array(interpolated_procrustes_curves)
for j in range(2):
    reference_curve = compute_frechet_mean(interpolated_procrustes_curves)
    temp_mean_shapes.append(reference_curve)
    interpolated_procrustes_curves = align_procrustes(interpolated_procrustes_curves, base_id=-2, external_curve=reference_curve)
    curve_functions = [parameterize_curve(curve) for curve in interpolated_procrustes_curves]

    reparam_curves = []
    optimal_path_dir = mkdir(bkup_dir, "optimal_path")
    for n in range(len(curve_functions)):
        func1 = curve_functions[n]
        cost_matrix = compute_cost_matrix(func1, reference_curve, t_value)
        optimal_path = find_optimal_reparametrization(cost_matrix)
        reparam_curve = reparameterize_curve(func1, optimal_path)
        reparam_curves.append(reparam_curve)
    reparam_curves = np.array(reparam_curves)
    frechet_reparam = compute_frechet_mean(reparam_curves)
    interpolated_procrustes_curves = reparam_curves

restore_pt_num = interpolate_pt_num//POINTS_NUM
Procrustes_curves = interpolated_procrustes_curves[:, ::restore_pt_num, :]
print ("Procrustes_curves.shape after alignment:", Procrustes_curves.shape)



frechet_curvature, frechet_torsion = compute_curvature_and_torsion(frechet_reparam)
frechet_energy = compute_geometry_param_energy(frechet_curvature, frechet_torsion)
frechet_tortuosity = compute_tortuosity(frechet_reparam)
print ("frechet_energy:", frechet_energy, "frechet_tortuosity:", frechet_tortuosity)
# fieldAttribute(list):
#  ['Curvature', 'float',pandas.Series],
#  ['Torsion', 'float',pandas.Series]]
scalarAttribute = [['Curvature', 'float',pd.Series(frechet_curvature)],
    ['Torsion', 'float',pd.Series(frechet_torsion)]]
# makeVtkFile(bkup_dir+"frechet_original.vtk", frechet_original, [], [])
makeVtkFile(bkup_dir+"frechet_reparam.vtk", frechet_reparam, scalarAttribute, [])

# dynamic time warping.
###################################

# log.write("- preprocessing_pca is not a SRVF PCA, and it is conducted on Procrustes_cruves before aligned_endpoints.\n")
# preprocessing_pca = PCAHandler(Procrustes_curves.reshape(len(Procrustes_curves),-1), None, 20, PCA_STANDARDIZATION)
# preprocessing_pca.PCA_training_and_test()
# preprocess_curves = preprocessing_pca.inverse_transform_from_loadings(preprocessing_pca.train_res).reshape(len(preprocessing_pca.train_res), -1, 3)

# Procrustes_curves = preprocess_curves

# log.write("Procrustes_curves is aligned again by endpoints.\n")
# Procrustes_curves = np.array([align_endpoints(curve, p) for curve in Procrustes_curves])

Curvatures, Torsions = compute_synthetic_curvature_and_torsion(Procrustes_curves)


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

LT_curvatures = Curvatures[LT_group]
HT_curvatures = Curvatures[HT_group]
print ("len(LT_curvatures):", len(LT_curvatures))
print ("len(HT_curvatures):", len(HT_curvatures))
average_of_LT_curvature = np.mean([np.mean(curv) for curv in LT_curvatures])
average_of_HT_curvature = np.mean([np.mean(curv) for curv in HT_curvatures])
# average_of_LT_curvature = np.mean([np.mean(curv) for curv in Curvatures])
# average_of_HT_curvature = np.mean([np.mean(curv) for curv in Curvatures])
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

# # 源文件和文件夹的路径
# source_code = 'D:/!BraVa_src/src/ModeDecomposition/Procrustes_tangentPCA.py'
# source_folder = 'D:/!BraVa_src/src/ModeDecomposition/myvtk'
# # 目标路径
# destination_code = os.path.join(bkup_dir, 'code.py')
# destination_folder = os.path.join(bkup_dir, 'myvtk')
# # 复制文件
# shutil.copy(source_code, destination_code)
# # 复制整个文件夹（包括内容）
# shutil.copytree(source_folder, destination_folder)

end_time = datetime.now()
total_time = end_time - start_time
print(dir_formatted_time, "is done in", total_time.seconds, "seconds.")
log.close()
open_folder_in_explorer(bkup_dir)
