import numpy as np
from scipy.stats import norm, entropy
from scipy.integrate import quad

# 假设的数据
length = 64
A_means = np.random.rand(length)
A_stds = np.random.rand(length)
B_means = np.random.rand(length)
B_stds = np.random.rand(length)
C_means = np.random.rand(length)
C_stds = np.random.rand(length)

# 计算两个正态分布的KL散度
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# 计算两个正态分布的Hellinger距离
def hellinger_distance(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

# 计算两个正态分布的总变差距离
def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

# 根据均值和标准差计算正态分布的概率
def calculate_probabilities(means, stds):
    return norm.pdf(np.arange(length), loc=means, scale=stds)

# A_probs = calculate_probabilities(A_means, A_stds)
# B_probs = calculate_probabilities(B_means, B_stds)
# C_probs = calculate_probabilities(C_means, C_stds)

# # 计算A和B之间的指标
# kl_ab = kl_divergence(A_probs, B_probs)
# hellinger_ab = hellinger_distance(A_probs, B_probs)
# total_variation_ab = total_variation_distance(A_probs, B_probs)

# # 计算A和C之间的指标
# kl_ac = kl_divergence(A_probs, C_probs)
# hellinger_ac = hellinger_distance(A_probs, C_probs)
# total_variation_ac = total_variation_distance(A_probs, C_probs)

# print(f"KL(A||B): {kl_ab}, Hellinger(A,B): {hellinger_ab}, TV(A,B): {total_variation_ab}")
# print(f"KL(A||C): {kl_ac}, Hellinger(A,C): {hellinger_ac}, TV(A,C): {total_variation_ac}")
