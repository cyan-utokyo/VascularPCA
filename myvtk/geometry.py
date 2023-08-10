import numpy as np
from scipy.signal import savgol_filter

def smooth_curve(curve, window_length=7, polyorder=3):
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
    
    return curvature, torsion

# curve = np.random.rand(64, 3)  # 用随机值作为例子
# curvature, torsion = compute_curvature_and_torsion(curve)
# print(curvature)
# print(torsion)


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