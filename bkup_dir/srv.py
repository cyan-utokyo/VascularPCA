import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

def square_root_velocity(trajectory):
    velocity = np.diff(trajectory, axis=0)
    velocity_magnitude =np.linalg.norm(velocity, axis=-1)
    for i in range(len(velocity)):
        # print (velocity[i])
        velocity[i,0]=velocity[i,0]/velocity_magnitude[i]
        velocity[i,1]=velocity[i,1]/velocity_magnitude[i]
        velocity[i,2]=velocity[i,2]/velocity_magnitude[i]
    return velocity




def f_transform_numpy(point, tol=1e-7):
    # 计算连续点之间的距离
    dists = np.sqrt(np.sum((point[..., 1:, :] - point[..., :-1, :]) ** 2, axis=-1))
    
    # 检查是否有重合的点
    if np.any(dists < tol):
        raise AssertionError(
            "The square root velocity framework "
            "is only defined for discrete curves "
            "with distinct consecutive sample points."
        )
    
    # 计算速度
    k_sampling_points = point.shape[1]
    coef = k_sampling_points - 1
    velocity = coef * (point[..., 1:, :] - point[..., :-1, :])
    
    # 计算速度的模长
    velocity_norm = np.sqrt(np.sum(velocity ** 2, axis=-1))
    
    # 计算平方根速度表示
    srv = velocity / np.sqrt(velocity_norm)[..., np.newaxis]
    
    # 处理多条曲线的情况
    n_points = point.shape[0]
    index = np.arange(n_points * k_sampling_points - 1)
    mask = ~((index + 1) % k_sampling_points == 0)
    srv = srv[mask]
    
    return srv