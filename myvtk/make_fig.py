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
import geomstats.geometry.discrete_curves as dc
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import HypersphereMetric
from scipy.spatial import distance
from myvtk.centerline_preprocessing import *
from scipy import interpolate
import matplotlib
import matplotlib.cm as cm
# from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from myvtk.dynamic_time_warp import *
from myvtk.Mypca import *
from myvtk.make_fig import *


def make_color(value_list, cmap="turbo"):
    # 使用你的数据 train_geodesic_d 创建一个颜色标准化器
    norm = plt.Normalize(min(value_list), max(value_list))
    # 使用标准化器和 colormap 创建一个 ScalarMappable
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    # 获取每一条线的颜色
    colors = sm.to_rgba(value_list)
    # test_line_colors = sm.to_rgba(test_geodesic_d)
    return colors