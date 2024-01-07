from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, chebyshev, cityblock, cosine, correlation, sqeuclidean
import numpy as np
from dtw import *
import numpy as np

# def calculate_dtw(x, y):
#     alignment = dtw(x, y, keep_internals=True, step_pattern=rabinerJuangStepPattern(4, "c"), 
#                     open_begin=True, open_end=True)
#     distance = alignment.distance
#     path = alignment.index1, alignment.index2
#     print("dist:", distance)
#     return distance, path

# def get_warping_function(path, x, y):
#     path_array = np.array(path).T
#     warp_function = np.zeros(len(x))
#     for i in range(path_array.shape[0]):
#         warp_function[path_array[i, 0]] = path_array[i, 1]
#     continuous_warp_function = np.interp(np.arange(x.shape[0]), path_array[:,0], path_array[:,1])
#     return continuous_warp_function

# mean_shape = np.mean(train_data_pre, axis=0)
# train_warping_functions = []
# for j in range(len(train_data_pre)):
#     distance, path = calculate_dtw(train_data_pre[j], mean_shape)
#     train_warping_functions.append(get_warping_function(path, train_data_pre[j], mean_shape))





def calculate_dtw(x, y):
    # distance, path = fastdtw(x, y, dist=euclidean)
    distance, path = fastdtw(x, y, dist=cosine)
    return distance, path

# distance, path = calculate_dtw(line_a, line_b)

def get_warping_function(path, x, y):
    path_array = np.array(path)
    warp_function = np.zeros(len(x))

    # 根据path定义warp function
    # 对于Q1中的每一个点，warp function的值就是与之对应的Q2中的点的索引
    for i in range(path_array.shape[0]):
        warp_function[path_array[i, 0]] = path_array[i, 1]

    # 因为warp function可能不是连续的，所以我们可以通过插值来得到一个连续的warp function
    # 这里我们用线性插值
    continuous_warp_function = np.interp(np.arange(x.shape[0]), path_array[:,0], path_array[:,1])
    return continuous_warp_function
    
