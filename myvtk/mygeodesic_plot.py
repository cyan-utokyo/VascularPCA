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
import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import DiscreteCurves
from geomstats.learning.frechet_mean import FrechetMean
import numpy as np
from scipy.interpolate import interp1d


import geomstats.geometry.discrete_curves as dc
import geomstats.backend as gs

def compute_geodesic_shapes_between_two_curves(curve_A, curve_B, num_steps=10):
    """
    Compute the shapes along the geodesic between two curves.

    Parameters:
        curve_A: np.array
            Starting curve (shape: [n_points, 3]).
        curve_B: np.array
            Ending curve (shape: [n_points, 3]).
        num_steps: int, optional
            Number of intermediate steps/shapes to compute.

    Returns:
        List of np.arrays, each of shape [n_points, 3].
    """
    # Create discrete curves space
    discrete_curves_space = dc.DiscreteCurves(ambient_manifold=dc.Euclidean(dim=3))
    
    # Ensure curves have the correct shape
    if curve_A.shape != curve_B.shape:
        raise ValueError("The dimensions of the two curves should be the same.")
    
    # Compute the geodesic path between the curves
    initial_tangent_vec = discrete_curves_space.metric.log(point=curve_B, base_point=curve_A)
    geodesic = discrete_curves_space.metric.geodesic(initial_point=curve_A, initial_tangent_vec=initial_tangent_vec)

    # Compute intermediate shapes along the geodesic
    t_values = gs.linspace(0., 1., num_steps)
    # geodesic_shapes = [geodesic(t) for t in t_values]
    geodesic_shapes = [gs.squeeze(geodesic(t)) for t in t_values]
    
    return np.array(geodesic_shapes)

def plot_curves_on_2d(curve_a, curve_b, geodesic_shapes, savepath):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)

    # Calculate offset
    offset = 0
    gap = 10  # Gap between curves

    # Plot curve_a
    ax.plot(np.arange(64) + offset, curve_a[:, 2], '-o', color='black', label="Seed")
    offset += 64 + gap

    # Plot geodesic shapes
    for i, shape in enumerate(geodesic_shapes):
        ax.plot(np.arange(64) + offset, shape[:, 2], '-o', color='blue', label=f"GS{i + 1}")
        offset += 64 + gap

    # Plot curve_b
    ax.plot(np.arange(64) + offset, curve_b[:, 2], '-o', color='red', label="Target")

    ax.invert_yaxis()
    ax.set_xticks([])  # remove x-ticks
    ax.set_yticks([])
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()