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
import geomstats.geometry.pre_shape as pre_shape
import geomstats.geometry.discrete_curves as dc
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import HypersphereMetric
from scipy.spatial import distance
from myvtk.centerline_preprocessing import *
from scipy import interpolate
import matplotlib
import matplotlib.cm as cm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from myvtk.customize_pca import *
from myvtk.make_fig import *
import shutil
import os
from myvtk.dtw import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter

ill=pd.read_csv("./illcases.txt",header=None)
ill = np.array(ill[0])
pre_files = glob.glob("./scaling/resamp_attr_ascii/vmtk64a/*.vtk")
unaligned_curves = []
Files = []
radii = []
Curvatures = []
Torsions = []
for idx in range(len(pre_files)):
    filename = pre_files[idx].split("\\")[-1].split(".")[0][:-8]
    if filename in ill:
        print (filename, "is found in illcases.txt, skip")
        continue
    # print (filename)
    pt, Curv, Tors, Radius, Abscissas, ptns, ftangent, fnormal, fbinormal = GetMyVtk(pre_files[idx], frenet=1)
    Files.append(pre_files[idx])
    unaligned_curves.append(pt)
    radii.append(Radius)
    Curvatures.append(Curv)
    Torsions.append(Tors)

unaligned_curves = np.array(unaligned_curves)
radii = np.array(radii)
Curvatures = np.array(Curvatures)
Torsions = np.array(Torsions)
Curvature_changes = np.diff(Curvatures, axis=1)
Torsion_changes = np.diff(Torsions, axis=1)

for i in range(len(Files)):
    if "BH0017_R" in Files[i]:
        base_id = i

print ("base_id:{},casename:{}で方向調整する".format(base_id, Files[base_id]))

a_curves = align_icp(unaligned_curves, base_id=base_id)
print ("First alignment done.")
Procrustes_curves = align_procrustes(a_curves,base_id=base_id)
print ("procrustes alignment done.")

parametrized_curves = np.zeros_like(Procrustes_curves)

for i in range(len(Procrustes_curves)):
    parametrized_curves[i] = arc_length_parametrize(Procrustes_curves[i])

Aligned_curves = align_curve(parametrized_curves)
Procrustes_curves = np.array(Procrustes_curves)

# 需要把长度对齐到原始曲线
for i in range(len(Aligned_curves)):
    aligned_length = measure_length(Aligned_curves[i])
    procrustes_length = measure_length(Procrustes_curves[i])
    Aligned_curves[i] = Aligned_curves[i] * (procrustes_length/aligned_length)

plt.plot(unaligned_curves[base_id][:,0], label="unaligned")
plt.plot(Procrustes_curves[base_id][:,0], label="procrustes")
plt.plot(Aligned_curves[base_id][:,0], label="aligned")
plt.legend()
plt.show()