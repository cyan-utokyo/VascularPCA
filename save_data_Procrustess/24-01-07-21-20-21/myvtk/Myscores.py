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
from scipy.spatial.distance import euclidean,pdist, squareform
from myvtk.Mypca import *
from myvtk.make_fig import *
import shutil
import os
from myvtk.dtw import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec
from sklearn.metrics import mutual_info_score
 
def compute_energy_centroid(energies):
    # Calculate the centroid of energy values.
    return np.mean(energies)

def compute_distance_from_centroid(energy, centroid):
    # Calculate the distance of an energy value from the centroid.
    return np.linalg.norm(np.array(energy) - np.array(centroid))

def score_energy(energy, centroid):
    # Define a scoring function. In this case, a lower distance means a higher score.
    # You may want to normalize or scale scores, depending on your application's needs.
    # Here we use a simple inverse relationship: score = 1 / (distance + epsilon)
    # to avoid division by zero.
    epsilon = 1e-6
    distance = compute_distance_from_centroid(energy, centroid)
    return 1 / (distance + epsilon)