import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
def smooth_curve(curve, window_length=5, polyorder=3):
    """
    平滑3D曲线使用Savitzky-Golay滤波器

    参数:
    - curve: 原始的3D曲线
    - window_length: 滤波器的窗口长度
    - polyorder: 多项式的阶数

    返回值:
    - 平滑后的3D曲线
    """
    # 确保window_length为奇数
    if window_length % 2 == 0:
        window_length += 1

    # 分别对x, y, z坐标进行平滑处理
    smoothed_curve = np.zeros_like(curve)
    for i in range(3):
        smoothed_curve[:, i] = savgol_filter(curve[:, i], window_length, polyorder)

    return smoothed_curve


import numpy as np

def compute_curvature_and_torsion(curve):
    window_size = 4
    # calculate moving averages using numpy convolve
    weights = np.repeat(1.0, window_size)/window_size

    # calculate first, second, and third derivatives using finite differences
    r_prime = np.diff(curve, axis=0)
    r_double_prime = np.diff(r_prime, axis=0)
    r_triple_prime = np.diff(r_double_prime, axis=0)

    # Pad derivatives to align array sizes
    r_prime = np.vstack((r_prime, np.zeros((1, 3))))
    r_double_prime = np.vstack((np.zeros((1, 3)), r_double_prime, np.zeros((1, 3))))
    r_triple_prime = np.vstack((np.zeros((2, 3)), r_triple_prime, np.zeros((2, 3))))

    # Ensure that r_prime, r_double_prime, and r_triple_prime have the same shape
    min_length = min(len(r_prime), len(r_double_prime), len(r_triple_prime))
    r_prime = r_prime[:min_length]
    r_double_prime = r_double_prime[:min_length]
    r_triple_prime = r_triple_prime[:min_length]

    cross_product = np.cross(r_prime, r_double_prime)
    cross_norm = np.linalg.norm(cross_product, axis=1)
    r_prime_norm = np.linalg.norm(r_prime, axis=1)

    epsilon = 1e-7
    curvature = np.where(r_prime_norm**3 > epsilon, cross_norm / (r_prime_norm ** 3), 0)

    torsion_numerator = np.einsum('ij,ij->i', r_prime, np.cross(r_double_prime, r_triple_prime))
    torsion = np.where(cross_norm**2 > epsilon, torsion_numerator / (cross_norm ** 2), 0)

    curvature = np.convolve(curvature, weights, 'valid')
    torsion = np.convolve(torsion, weights, 'valid')

    # Apply non-linear transformation
    curvature = np.tanh(curvature)
    torsion = np.tanh(torsion)

    # Create the interpolator functions for curvature and torsion
    # We are using 'linear' interpolation and 'extrapolate' to allow extension beyond the original range
    interp_curvature = interp1d(np.arange(len(curvature)), curvature, kind='linear', fill_value="extrapolate")
    interp_torsion = interp1d(np.arange(len(torsion)), torsion, kind='linear', fill_value="extrapolate")

    # Use the interpolator functions to extend the arrays to the original curve length
    interpolated_curvature = interp_curvature(np.arange(len(curve)))
    interpolated_torsion = interp_torsion(np.arange(len(curve)))

    return interpolated_curvature, interpolated_torsion


def compute_synthetic_curvature_and_torsion(C_recovered):
    C_srvf_synthetic_curvature = []
    C_srvf_synthetic_torsion = []
    for i in range(len(C_recovered)):
        C_srvf_synthetic_curvature.append(compute_curvature_and_torsion(C_recovered[i])[0])
        C_srvf_synthetic_torsion.append(compute_curvature_and_torsion(C_recovered[i])[1])
    C_srvf_synthetic_curvature = np.array(C_srvf_synthetic_curvature)
    C_srvf_synthetic_torsion = np.array(C_srvf_synthetic_torsion)
    return C_srvf_synthetic_curvature, C_srvf_synthetic_torsion


def compute_geometry_param_energy(curvature, torsion, POWER_ENG_CURVATURE=2, POWER_ENG_TORSION=2):
    # adjusted_torsion = np.tanh(torsion) * 0.5 + 0.5
    curvature_energy = np.mean(np.power(curvature, POWER_ENG_CURVATURE))
    torsion_energy = np.mean(np.power(torsion, POWER_ENG_TORSION))
    return curvature_energy,torsion_energy

# def compute_geometry_param_energy(curvature, torsion, POWER_ENG_CURVATURE=2, POWER_ENG_TORSION=2):
#      """
#      这个函数并未被使用，会让torsion失去分辨力。
#      """
#     # 使用简单的前向差分核
#     kernel = [1, -1]
    
#     # 对曲率和扭率应用卷积
#     conv_curvature = np.convolve(curvature, kernel, mode='valid')
#     conv_torsion = np.convolve(torsion, kernel, mode='valid')
    
#     # 计算卷积后的能量
#     curvature_energy = np.mean(np.power(conv_curvature, POWER_ENG_CURVATURE))
#     torsion_energy = np.mean(np.power(conv_torsion, POWER_ENG_TORSION))
    
#     return curvature_energy, torsion_energy


def compute_tortuosity(curve):
    """
    计算曲线的扭曲度
    """
    # 计算曲线的长度
    curve_length = np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
    # 计算曲线的欧式距离
    euclidean_distance = np.linalg.norm(curve[-1] - curve[0])
    # 计算扭曲度
    tortuosity = curve_length / euclidean_distance
    return tortuosity


def compute_geometry_param_energy_segment(curvature, torsion, POWER_ENG_CURVATURE=2, POWER_ENG_TORSION=2):
    """
    这个函数并未被使用。它计算曲率和扭率的能量，但是将曲线分成了几个段，然后计算每个段的能量。
    """
    #  分段节点
    nodes = [0, 6, 25, 44, 54, len(curvature)]  # 添加0和len(curvature)以确保包括所有数据点
    # 初始化存储结果的列表
    curvature_energies = []
    torsion_energies = []

    # 遍历每个分段
    for i in range(len(nodes) - 1):
        start, end = nodes[i], nodes[i+1]
        # 计算当前分段的curvature和torsion能量
        curvature_energy = np.mean(np.power(curvature[start:end], POWER_ENG_CURVATURE))
        torsion_energy = np.mean(np.power(torsion[start:end], POWER_ENG_TORSION))
        # 将结果添加到列表中
        curvature_energies.append(curvature_energy)
        torsion_energies.append(torsion_energy)

    return curvature_energies, torsion_energies