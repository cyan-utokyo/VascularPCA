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
import seaborn as sns
import geomstats.geometry.discrete_curves as dc
import geomstats.backend as gs

def compute_geodesic_shapes_between_two_curves(curve_A, curve_B, num_steps=10, initial_tangent_vec=None):
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
    ambient_manifold = dc.Euclidean(dim=3)
    srv_metric = dc.SRVMetric(ambient_manifold=ambient_manifold)
    discrete_curves_space = dc.DiscreteCurves(ambient_manifold=ambient_manifold, metric=srv_metric)
    
    
    # Ensure curves have the correct shape
    if curve_A.shape != curve_B.shape:
        raise ValueError("The dimensions of the two curves should be the same.")
    
    # Compute the geodesic path between the curves
    if initial_tangent_vec is None:
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
    ax.plot(np.arange(64) + offset, curve_a[:, 2], linewidth=8,color='black', label="Seed")
    offset += 64 + gap

    # Generate the color palette for geodesic shapes
    colors = sns.cubehelix_palette(n_colors=len(geodesic_shapes), start=.5, rot=-.5)

    # Plot geodesic shapes
    for i, shape in enumerate(geodesic_shapes):
        ax.plot(np.arange(64) + offset, shape[:, 2], linewidth=4, color=colors[i], label=f"GS{i + 1}")
        offset += 64 + gap

    # Plot curve_b
    ax.plot(np.arange(64) + offset, curve_b[:, 2], linewidth=8,  color='red', label="Target")

    ax.invert_yaxis()
    ax.set_xticks([])  # remove x-ticks
    ax.set_yticks([])
    # ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()



def plot_curvature_torsion_heatmaps(curvature, torsion, savepath):
    fig = plt.figure(dpi=300, figsize=(10, 5))
    gs = fig.add_gridspec(6, 2)  # Create a grid of 6 rows by 2 columns

    ax1 = fig.add_subplot(gs[:5, 0])  # Allocate 5 rows for heatmap1
    ax2 = fig.add_subplot(gs[:5, 1])  # Allocate 5 rows for heatmap2

    # Define colorbar locations for each heatmap
    cbar_ax1 = fig.add_axes([0.15, 0.08, 0.3, 0.02])
    cbar_ax2 = fig.add_axes([0.55, 0.08, 0.3, 0.02])

    # Draw heatmaps with separate colorbars
    sns.heatmap(curvature, cmap="YlGnBu", ax=ax1, linewidths=0.1, linecolor='white', vmin=0, vmax=1.2,
                cbar_ax=cbar_ax1, cbar_kws={"orientation": "horizontal"})
    sns.heatmap(torsion, cmap="YlGnBu", ax=ax2, linewidths=0.1, linecolor='white', vmin=-1.2, vmax=1.2, 
                cbar_ax=cbar_ax2, cbar_kws={"orientation": "horizontal"})

    # Set titles
    ax1.set_title("Curvature")
    ax2.set_title("Torsion")

    # Adjust y-axis ticks
    ax1.set_yticks([0, curvature.shape[0]-1])
    ax1.set_yticklabels(["seed", "target"])
    ax2.set_yticks([0, torsion.shape[0]-1])
    ax2.set_yticklabels(["seed", "target"])

    # Adjust x-axis ticks and their orientation
    ax1.set_xticks([0, curvature.shape[1]-1])
    ax1.set_xticklabels(["proximal", "distal"], rotation=0)
    ax2.set_xticks([0, torsion.shape[1]-1])
    ax2.set_xticklabels(["proximal", "distal"], rotation=0)

    # Save figure
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

# Example usage:
# plot_curvature_torsion_heatmaps(geod_curvature, geod_torsion, "path_to_save.png")
