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
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import geomstats.geometry.discrete_curves as dc
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import HypersphereMetric
from scipy.spatial import distance
# 创建一个新的目录来保存变换后的曲线

def align_procrustes(curves,base_id=0):
    # 使用第一条曲线作为基准曲线
    base_curve = curves[base_id]
    new_curves = []
    
    for i in range(len(curves)):
        curve = curves[i]
        # print (orthogonal(base_curve, curve, translate=True, scale=True))
        # 对齐当前曲线到基准曲线
        result = orthogonal(base_curve, curve, translate=True, scale=False)
    
        # Z是Procrustes变换后的曲线，用它来更新原始曲线
        # curves[i] = Z
        new_curves.append(result['new_b'])

    return np.array(new_curves)

def arc_length_parametrize(curve):
    # 计算曲线上每两个连续点之间的距离
    diffs = np.diff(curve, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))

    # 累加这些距离得到每个点的弧长
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

    # 生成新的参数值（等间距）
    num_points = len(curve)
    new_params = np.linspace(0, arc_lengths[-1], num_points)

    # 对曲线的每一个维度进行插值
    new_curve = np.zeros_like(curve)
    for i in range(3):
        interp = interp1d(arc_lengths, curve[:, i], kind='cubic')
        new_curve[:, i] = interp(new_params)

    return new_curve

def calculate_srvf(curve):
    # 计算曲线的导数
    t = np.linspace(0, 1, curve.shape[0])
    cs = CubicSpline(t, curve)
    derivative = cs(t, 1)
    
    # 计算SRVF
    magnitude = np.linalg.norm(derivative, axis=1)
    eps = 1e-8  # add a small positive number to avoid division by zero
    srvf = np.sqrt(magnitude + eps)[:, np.newaxis] * derivative / (magnitude[:, np.newaxis] + eps)
    return srvf

def inverse_srvf(srvf, initial_point):
    # print ("calling inverse_srvf, magnitude = np.linalg.norm(srvf, axis=1)")
    # 计算速度
    # magnitude = np.linalg.norm(srvf, axis=1)
    magnitude = np.linalg.norm(srvf, axis=1) ** 2
    velocity = srvf / np.sqrt(magnitude[:, np.newaxis])

    # 对速度进行积分
    curve = np.cumsum(velocity, axis=0)
    
    # 将曲线的初始点设为给定的初始点
    curve = curve - curve[0] + initial_point

    return curve

def align_curve(curve):
    # 将曲线平移到原点
    # print ("debug align_curve")
    # print ("curve.shape:", curve.shape)
    curve_centered = curve - np.mean(curve, axis=0)

    # Reshape the centered curve into 2D array
    curve_centered_2d = curve_centered.reshape(-1, curve_centered.shape[1]*curve_centered.shape[2])

    # 计算PCA
    pca = PCA(n_components=50)
    curve_pca = pca.fit_transform(curve_centered_2d)
    curve_pca = pca.inverse_transform(curve_pca)
    # print ("curve_pca.shape:", curve_pca.shape)

    # reshape back to the original shape
    curve_pca = curve_pca.reshape(curve_centered.shape)

    return curve_pca

def align_icp(curves,base_id=80):
    # 使用第一条曲线作为基准曲线
    new_curves = []
    if base_id > -1:
        base_curve = curves[base_id]
    elif base_id == -1:
        base_curve = np.mean(curves, axis=0)
    
    for i in range(len(curves)):
        curve = curves[i]
        
        # 计算基准曲线和当前曲线的质心
        base_centroid = np.mean(base_curve, axis=0)
        curve_centroid = np.mean(curve, axis=0)
        
        # 将两个曲线移到原点
        base_centered = base_curve - base_centroid
        curve_centered = curve - curve_centroid
        
        # 计算最优旋转
        H = np.dot(base_centered.T, curve_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
           Vt[-1,:] *= -1
           R = np.dot(Vt.T, U.T)
        
        # 旋转和平移当前曲线以对齐到基准曲线
        # curves[i] = np.dot((curve - curve_centroid), R.T) + base_centroid
        new_curves.append(np.dot((curve - curve_centroid), R.T) + base_centroid)

    return np.array(new_curves)

def compute_geodesic_dist(curves):
    geodesic_d = []
    curve_A = np.mean(curves, axis=0)
    for idx in range(len(curves)):
        curve_B = curves[idx]
        # 创建离散曲线对象
        discrete_curves_space = dc.DiscreteCurves(ambient_manifold=dc.Euclidean(dim=3))
        # 计算测地线距离
        geodesic_distance = discrete_curves_space.metric.dist(curve_A, curve_B)
        geodesic_d.append(geodesic_distance)
    return np.array(geodesic_d)

def recovered_curves(inverse_data, is_srvf):
    recovered_curves = []
    if is_srvf:
        for cv in inverse_data:
            recovered_curves.append(inverse_srvf(cv, np.zeros(3)))
    else:
        recovered_curves = inverse_data
    return np.array(recovered_curves)

def write_curves_to_vtk(curves, files, dir):
    for i in range(len(curves)):
        filename = files[i].split("\\")[-1].split(".")[0]
        makeVtkFile(dir+"{}.vtk".format(filename), curves[i], [],[])

def process_data_key(data_key, train_inverse, test_inverse, train_files, test_files, inverse_data_dir):
    is_srvf = "SRVF" in data_key
    train_inverse = recovered_curves(train_inverse, is_srvf)
    test_inverse = recovered_curves(test_inverse, is_srvf)

    inverse_dir = mkdir(inverse_data_dir, "coords_" + data_key)
    write_curves_to_vtk(train_inverse, train_files, inverse_dir + "train_")
    write_curves_to_vtk(test_inverse, test_files, inverse_dir + "test_")

    # print (train_inverse.shape, test_inverse.shape)



