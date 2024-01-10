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


def fit_gaussian(data):
    # 计算数据的均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # 创建一个高斯分布对象数组
    gaussians = [norm(loc=mean[i], scale=std[i]) for i in range(len(mean))]
    return gaussians

def from_tangentPCA_feature_to_curves(tpca, tangent_base, tangent_projected_data, PCA_N_COMPONENTS, discrete_curves_space, inverse_srvf_func=None):
    principal_components = tpca.components_
    # Assuming principal_components has the shape (n_components, n_sampling_points * n_dimensions)
    point_num = len(tangent_base)
    principal_components_reshaped = principal_components.reshape((PCA_N_COMPONENTS, point_num, 3)) # point_num是采样点数
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
        reconstructed_curve = discrete_curves_space.metric.exp(
            tangent_vec=tangent_vector_reconstructed, base_point=tangent_base
        )
        # reconstructed_curve = inverse_srvf(reconstructed_srvf, np.zeros(3))
        # print ("reconstructed_curve length:", measure_length(reconstructed_curve))# length=63
        
        reconstructed_curves.append(reconstructed_curve)
    reconstructed_curves = np.array(reconstructed_curves)
    return reconstructed_curves


def reconstruct_components(tpca, discrete_curves_space, tangent_base, inverse_srvf_func=None):
    principal_components = tpca.components_
    point_num = len(tangent_base)
    # Assuming the shape of principal_components is (n_components, n_sampling_points * n_dimensions)
    principal_components_reshaped = principal_components.reshape(
        (tpca.n_components, point_num, 3)
    )

    curves_from_components = []
    for component in principal_components_reshaped:
        # Map the tangent vector back to the curve space using the exponential map.
        curve = discrete_curves_space.metric.exp(
            tangent_vec=component, base_point=tangent_base
        )
        # Apply the inverse SRVF to get the curve
        # curve = inverse_srvf_func(srvf_curve, np.zeros(3))  # Assuming inverse_srvf function takes srvf_curve and a base point as input.
        curves_from_components.append(curve)
    #curves_from_components = align_icp(curves_from_components, base_id=0)
    #print ("First alignment done.")
    #curves_from_components = align_procrustes(curves_from_components,base_id=0)
    # Visualize each reconstructed component curve
    return curves_from_components