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
    # test_warp_function_list = []
    # test_transformed_Q1_list = []

    # Step 1: 对SRVF进行动态时间规整
    # distance, path = fastdtw(Q1, Q2, dist=euclidean)

    # Step 2: 计算最优变换
    # path是一个包含配对索引的列表，我们可以使用它来创建变换后的曲线
    # 我们假设Q1是我们想要对齐的曲线，Q2是参考曲线
    # test_transformed_Q1 = np.zeros(Q1.shape)
    # test_transformed_L1 = np.zeros(Q1.shape)
        
    # for pair in path:
    #     test_transformed_Q1[pair[0]] = Q2[pair[1]] # 原本是Q1[pair[1]]
    #     test_transformed_L1[pair[0]] = train_mean_curve[pair[1]] # 原本是Q1[pair[1]]

    # 这个transformed_Q1现在是变换后的SRVF，你可以将其转换回原始曲线空间
    # 将path转化为numpy array，以便于处理
    path_array = np.array(path)

    # 初始化warp function为一个与Q1同样长度的零向量
    warp_function = np.zeros(len(x))

    # 根据path定义warp function
    # 对于Q1中的每一个点，warp function的值就是与之对应的Q2中的点的索引
    for i in range(path_array.shape[0]):
        warp_function[path_array[i, 0]] = path_array[i, 1]

    # 因为warp function可能不是连续的，所以我们可以通过插值来得到一个连续的warp function
    # 这里我们用线性插值
    continuous_warp_function = np.interp(np.arange(x.shape[0]), path_array[:,0], path_array[:,1])
    return continuous_warp_function
    
