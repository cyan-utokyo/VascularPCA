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

# 使用您的曲线作为例子
# curve = np.random.rand(64, 3)  # 用随机值作为例子
# smoothed_curve = smooth_curve(curve)
# curvature, torsion = compute_curvature_and_torsion(smoothed_curve)
# print(curvature)
# print(torsion)

def compute_curvature_and_torsion(curve):
    # 使用有限差分计算第一、第二和第三导数
    curve =smooth_curve(curve)
    r_prime = np.diff(curve, axis=0, n=1)
    r_double_prime = np.diff(curve, axis=0, n=2)
    r_triple_prime = np.diff(curve, axis=0, n=3)
    
    # 为了对齐数组大小，我们在开始和/或结束时加上零行
    r_prime = np.vstack((r_prime, np.zeros((1, 3))))
    r_double_prime = np.vstack((np.zeros((1, 3)), r_double_prime, np.zeros((1, 3))))
    r_triple_prime = np.vstack((np.zeros((2, 3)), r_triple_prime, np.zeros((2, 3))))

    # 这里，我们需要确保r_prime, r_double_prime和r_triple_prime具有相同的形状
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

    curvature = np.tanh(curvature)
    torsion = np.tanh(torsion)
    
    return curvature, torsion

# curve = np.random.rand(64, 3)  # 用随机值作为例子
# curvature, torsion = compute_curvature_and_torsion(curve)
# print(curvature)
# print(torsion)

def compute_synthetic_curvature_and_torsion(C_recovered, weights):
    C_srvf_synthetic_curvature = []
    C_srvf_synthetic_torsion = []
    for i in range(len(C_recovered)):
        C_srvf_synthetic_curvature.append(np.convolve(compute_curvature_and_torsion(C_recovered[i])[0], weights, 'valid'))
        C_srvf_synthetic_torsion.append(np.convolve(compute_curvature_and_torsion(C_recovered[i])[1], weights, 'valid'))
    C_srvf_synthetic_curvature = np.array(C_srvf_synthetic_curvature)
    C_srvf_synthetic_torsion = np.array(C_srvf_synthetic_torsion)
    return C_srvf_synthetic_curvature, C_srvf_synthetic_torsion

from scipy.signal import find_peaks

def calculate_scores(curvature, torsion):
    # 初始化分数
    scores = {"C": 0, "U": 0, "S": 0, "V": 0}
    
    # 1. 找到曲率的峰值
    peak_indices, peak_properties = find_peaks(curvature, height=0.1) # 0.1是阈值，需要根据实际数据调整

    # 如果只有一个峰值（ica siphon），并且没有其他显著峰值，可能是V型
    if len(peak_indices) == 1:
        scores["V"] += 1

    # 2. 检查torsion变化
    if peak_indices:
        max_peak_index = peak_indices[np.argmax(peak_properties["peak_heights"])]
        torsion_change = abs(torsion[max_peak_index+1] - torsion[max_peak_index-1])
        if torsion_change > 0.1:  # 0.1是一个阈值，需要根据实际数据调整
            scores["C"] += 1

    # 3. 检查多个曲率峰值
    if len(peak_indices) > 1:
        primary_peak = peak_properties["peak_heights"][0]
        for i in range(1, len(peak_properties["peak_heights"])):
            if abs(primary_peak - peak_properties["peak_heights"][i]) < 0.05:  # 0.05是一个阈值，需要调整
                scores["S"] += 1
                break

    # 4. 如果没有明确的特点，可能是U型
    if max(scores.values()) == 0:
        scores["U"] = 1
        
    return scores

def determine_type(curve):
    curvature, torsion = compute_curvature_and_torsion(curve)
    scores = calculate_scores(curvature, torsion)
    return max(scores, key=scores.get)


def plot_curves_with_peaks(i, j, Procrustes_curves, Curvatures, Torsion, savepath, axes=(0,1), distance=5):
    # 获取峰值的位置
    peaks_curvature_i = find_peaks(Curvatures[i], height=0.25, distance=5)[0]
    peaks_curvature_j = find_peaks(Curvatures[j], height=0.25, distance=5)[0]
    peaks_torsion_i = find_peaks(Torsion[i])[0]
    peaks_torsion_j = find_peaks(Torsion[j])[0]

    fig, ax = plt.subplots()
    # 绘制3D曲线在2D平面上
    ax.plot(Procrustes_curves[i, :, axes[0]], Procrustes_curves[i, :, axes[1]], color='k', label='Curve ' + str(i))
    ax.plot(Procrustes_curves[j, :, axes[0]] + distance, Procrustes_curves[j, :, axes[1]], color='k', label='Curve ' + str(j))  

    # 添加峰值的点
    ax.scatter(Procrustes_curves[i, peaks_curvature_i, axes[0]], Procrustes_curves[i, peaks_curvature_i, axes[1]], color='red', s=50, label='Curvature Peaks i')
    ax.scatter(Procrustes_curves[j, peaks_curvature_j, axes[0]] + distance, Procrustes_curves[j, peaks_curvature_j, axes[1]], color='red', s=50, label='Curvature Peaks j')
    # ax.scatter(Procrustes_curves[i, peaks_torsion_i, axes[0]], Procrustes_curves[i, peaks_torsion_i, axes[1]], color='blue', s=50, label='Torsion Peaks i')
    # ax.scatter(Procrustes_curves[j, peaks_torsion_j, axes[0]] + distance, Procrustes_curves[j, peaks_torsion_j, axes[1]], color='blue', s=50, label='Torsion Peaks j')

    # 连接红点
    for p1, p2 in zip(peaks_curvature_i, peaks_curvature_j):
        ax.plot([Procrustes_curves[i, p1, axes[0]], Procrustes_curves[j, p2, axes[0]] + distance], [Procrustes_curves[i, p1, axes[1]], Procrustes_curves[j, p2, axes[1]]], 'r-', lw=1)
    # 连接蓝点
    # for p1, p2 in zip(peaks_torsion_i, peaks_torsion_j):
    #     ax.plot([Procrustes_curves[i, p1, axes[0]], Procrustes_curves[j, p2, axes[0]] + distance], [Procrustes_curves[i, p1, axes[1]], Procrustes_curves[j, p2, axes[1]]], 'b-', lw=1)

    # ax.legend()
    plt.savefig(savepath)
    plt.close()

# # 示例
# i, j = 0, 1  # 按需要更改
# plot_curves_with_peaks(i, j, Procrustes_curves, Curvatures, Torsion)


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