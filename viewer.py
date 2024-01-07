import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonDataModel import vtkPolyData
# from vtkmodules.vtkFiltersModeling import vtkTubeFilter
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from myvtk.geometry import compute_curvature_and_torsion
from myvtk.centerline_preprocessing import compute_frechet_mean
from myvtk.GetMakeVtk import measure_length
from scipy.interpolate import interp1d
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import dill as pickle

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

POINTS_NUM =64
tangent_projected_data = np.load("tangent_projected_data.npy")
Procrustes_curves = np.load("Procrustes_curves.npy")
tpca = pickle.load(open("tpca.pkl", "rb"))
r3 = Euclidean(dim=3)
srv_metric = SRVMetric(r3)
discrete_curves_space = DiscreteCurves(ambient_manifold=r3, k_sampling_points=POINTS_NUM)
Procs_srvf_curves = np.zeros_like(Procrustes_curves)
for i in range(len(Procrustes_curves)):
    # Procs_srvf_curves[i] = calculate_srvf((Procrustes_curves[i])/measure_length(Procrustes_curves[i]))
    # Procs_srvf_curves[i] = calculate_srvf((Procrustes_curves[i])/np.linalg.norm(Procrustes_curves[i][-1] - Procrustes_curves[i][0]))
    Procs_srvf_curves[i] =Procrustes_curves[i]/measure_length(Procrustes_curves[i])
tangent_base = compute_frechet_mean(Procs_srvf_curves)
frechet_mean_shape = compute_frechet_mean(Procs_srvf_curves)
tangent_vectors = []
for curve in Procs_srvf_curves:
    tangent_vector = discrete_curves_space.to_tangent(curve, tangent_base)
    tangent_vectors.append(tangent_vector)
tangent_vectors = np.array(tangent_vectors)
tpca.transform(tangent_vectors)

def length_normalize(curve):
    return curve/measure_length(curve)

def tangent_mapping(curve,tangent_base):
    tangent_vector = discrete_curves_space.to_tangent(curve, tangent_base)
    return tangent_vector

# 假设我们关注的是点的y坐标
def extract_slider_values_from_vtk(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    poly_data = reader.GetOutput()
    points = poly_data.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    values = []
    for i in range(num_points):
        x, y, z = points.GetPoint(i)
        values.append(y)  # 假设我们使用y坐标
    
    return values

def float_to_int(value, min_float, max_float, max_int):
    # 将浮点数映射到整数
    return int((value - min_float) / (max_float - min_float) * max_int)

def int_to_float(value, min_float, max_float, max_int):
    # 将整数映射回浮点数
    return min_float + (value / max_int) * (max_float - min_float)

def compute_curvature_and_torsion_vtk(pts):
    if measure_length(pts) < 10:
        return np.zeros(len(pts)), np.zeros(len(pts))
    # interpolate the points to have a fixed number of points
    num_points = 100
    t = np.linspace(0, 1, len(pts))
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    fx = interp1d(t, x, kind='cubic')
    fy = interp1d(t, y, kind='cubic')
    fz = interp1d(t, z, kind='cubic')
    new_t = np.linspace(0, 1, num_points)
    x = fx(new_t)
    y = fy(new_t)
    z = fz(new_t)
    new_pts = np.array([x, y, z]).T
    curvature, torsion = compute_curvature_and_torsion(new_pts)
    return curvature, torsion


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # 创建中央窗口部件
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建水平布局
        self.horizontal_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # 存储所有滑块值的列表
        self.slider_values = [5] * 8  # 假设初始值为5

        # 创建左侧滑块的网格布局
        self.left_panel = QtWidgets.QGridLayout()  # 使用 QGridLayout
        self.sliders = []
        self.slider_labels = []

        for i in range(8):
            # 创建滑块
            # slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            # 设置滑块的属性，例如范围等

            # 创建对应的标签
            label = QtWidgets.QLabel('0.00 - 1.00')  # 初始标签文本
            # 设置标签的其他属性，如对齐、字体等

            # 将滑块和标签添加到左侧面板布局中
            self.left_panel.addWidget(label, i, 0)  # 将标签添加到布局中
            self.left_panel.addWidget(slider, i, 1)  # 将滑块添加到布局中

            # 添加到滑块和标签列表中
            self.sliders.append(slider)
            self.slider_labels.append(label)

        # 创建右侧图表面板
        self.right_panel = QtWidgets.QVBoxLayout()
        self.figures = []
        self.axes = []
        for _ in range(2):
            fig = Figure(figsize=(5, 3))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.plot(self.slider_values, color="dimgray")  # 使用滑块值初始化图表
            ax.set_xticklabels([])
            self.figures.append(fig)
            self.axes.append(ax)
            self.right_panel.addWidget(canvas)
        
        # 创建VTK渲染窗口
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        
        # 将面板和VTK窗口添加到布局中
        self.horizontal_layout.addLayout(self.left_panel)
        self.horizontal_layout.addWidget(self.vtk_widget)
        self.horizontal_layout.addLayout(self.right_panel)

        # 配置滑块
        self.configure_sliders()
        
        # 添加菜单栏的 'Open' 操作
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu('&File')
        open_action = file_menu.addAction('&Open')
        open_action.triggered.connect(self.open_file)

        # 初始化VTK场景（没有对象）
        self.initialize_vtk_scene()
        self.original_poly_data = None

    def configure_sliders(self):
        # 配置所有滑块
        for slider in self.sliders:
            slider.setMinimum(0)
            slider.setMaximum(10)
            slider.setValue(5)
            slider.valueChanged.connect(self.on_slider_value_changed)


    def initialize_vtk_scene(self):
        self.renderer = vtk.vtkRenderer()

        # 创建坐标轴指示器
        axes = vtk.vtkAxesActor()
        axes.SetAxisLabels(0)  # 不显示坐标轴标签

        # 添加坐标轴指示器到渲染器
        self.renderer.AddActor(axes)

        self.renderer.AddActor(axes)
        # 初始化 VTK 视图窗口
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.Initialize()


    def open_file(self):
        # Open file dialog and select .vtk file
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "VTK Files (*.vtk)",
            options=options
        )
        if file_name:
            self.display_polyline(file_name)
            slider_values = extract_slider_values_from_vtk(file_name)
            self.initialize_sliders(slider_values)


    def on_slider_value_changed(self, value):
        # 当滑块值改变时调用
        min_float, max_float = 0.0, 1.0
        max_int = 1000  # 滑块的最大值
        float_value = int_to_float(value, min_float, max_float, max_int)
        print("Slider value changed to:", float_value)
        self.adjust_vtk_shape()
        self.recompute_and_update_plots()

    def display_polyline(self, file_name):
        # Read the .vtk file containing the polyline
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()

        # 保存原始的 vtkPolyData 对象
        self.original_poly_data = reader.GetOutput()  # 添加这一行
        # 创建原始数据的深拷贝
        self.original_poly_data_copy = vtk.vtkPolyData()
        self.original_poly_data_copy.DeepCopy(self.original_poly_data)

        # 使用 vtkOutlineFilter 为对象添加边框
        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(reader.GetOutputPort())

        # 创建 mapper 和 actor 用于显示边框
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())

        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)

        # 添加边框 actor 到渲染器
        self.renderer.AddActor(outline_actor)

        # 提取点数据
        pts = vtk_to_numpy(self.original_poly_data.GetPoints().GetData())
        tangent_pt = tangent_mapping(length_normalize(pts),tangent_base)
        
        curvature, torsion = compute_curvature_and_torsion_vtk(pts)
        # 使用曲率和挠率数据更新右侧图表
        self.update_plots(curvature, torsion)
        
        self.pts = pts
        self.curvature = curvature
        self.torsion = torsion
        self.tangent_pt = tangent_pt
        print ("self.tangent_pt:", self.tangent_pt)
        

        # Create a mapper and actor for the polyline
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Add the actor to the scene and render
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()  # Adjust the camera to see the object
        self.vtk_widget.GetRenderWindow().Render()


    def update_sliders(self, idx, value):
        # 更新滑块值
        self.slider_values[idx] = value

    def initialize_sliders(self, slider_values):
        # 假设滑块表示的浮点数范围是0到1
        min_values = np.min(tangent_projected_data[:, :8], axis=0)
        max_values = np.max(tangent_projected_data[:, :8], axis=0)

        # 初始化滑块和标签
        for i, slider in enumerate(self.sliders):
            min_float = min_values[i]
            max_float = max_values[i]
            max_int = 1000  # 滑块的整数表示的最大值

            # 设置滑块的最小值和最大值
            slider.setMinimum(float_to_int(min_float, min_float, max_float, max_int))
            slider.setMaximum(float_to_int(max_float, min_float, max_float, max_int))

            # 添加标签以显示最大值和最小值
            label_text = f"{min_float:.2f} - {max_float:.2f}"
            self.slider_labels[i].setText(label_text)
    def update_plots(self, curvature, torsion):
            # 假设我们有两个图表，一个用于曲率，一个用于挠率
            self.axes[0].clear()
            self.axes[0].plot(curvature, color="dimgray")
            self.axes[0].set_title("Curvature")
            self.axes[0].set_xticklabels([])

            self.axes[1].clear()
            self.axes[1].plot(torsion, color="dimgray")
            self.axes[1].set_title("Torsion")
            self.axes[1].set_xticklabels([])

            for fig in self.figures:
                fig.canvas.draw()

    def adjust_vtk_shape(self):
        print("adjust_vtk_shape")
        if self.original_poly_data_copy is None:
            return

        # 获取当前所有滑块的值
        slider_values = [slider.value() for slider in self.sliders]

        # 使用原始数据的副本作为基础来调整形状
        new_points = vtk_to_numpy(self.original_poly_data_copy.GetPoints().GetData()).copy()
        for i, displacement in enumerate(slider_values):
            if i < len(new_points):
                new_points[i, 0] += displacement  # 假设改变的是 x 方向

        # 更新 vtkPolyData
        modified_points = numpy_to_vtk(new_points)
        self.original_poly_data.GetPoints().SetData(modified_points)
        self.original_poly_data.GetPoints().Modified()

        # 重新渲染
        self.vtk_widget.GetRenderWindow().Render()

    def recompute_and_update_plots(self):
        
        if self.original_poly_data is None:
            return
        print ("recompute_and_update_plots")
        # 从 vtkPolyData 提取点
        pts = vtk_to_numpy(self.original_poly_data.GetPoints().GetData())

        # 重新计算曲率和挠率
        curvature, torsion = compute_curvature_and_torsion_vtk(pts)

        # 更新图表
        self.update_plots(curvature, torsion)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())