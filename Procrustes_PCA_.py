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
from scipy.spatial.distance import euclidean
from myvtk.customize_pca import *
from myvtk.make_fig import *
import shutil
import os
from myvtk.dtw import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec
from myvtk.scores import *
import csv
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal, kde
import seaborn as sns
import copy



SCALETO1 = True
log = open("./log.txt", "w")
# 获取当前时间
start_time = datetime.now()
smooth_scale = 0.01
# 将时间格式化为 'yymmddhhmmss' 格式
dir_formatted_time = start_time.strftime('%y-%m-%d-%H-%M-%S')
log.write("Start at: {}\n".format(dir_formatted_time))
bkup_dir = mkdir("./", "save_data_Procrustess")
bkup_dir = mkdir(bkup_dir, dir_formatted_time)
current_file_path = os.path.abspath(__file__)
current_file_name = os.path.basename(__file__)
backup_file_path = os.path.join(bkup_dir, current_file_name)
shutil.copy2(current_file_path, backup_file_path)

print ("backup dir: ", bkup_dir)
# shutil.copyfile(source_file, destination_file)
log = open(bkup_dir+"log.txt", "w")
# 创建一个新的目录来保存变换后的曲线
cmap = matplotlib.cm.get_cmap('RdGy')


ill=pd.read_csv("./illcases.txt",header=None)
ill = np.array(ill[0])
print (ill)
pre_files = glob.glob("./scaling/resamp_attr_ascii/vmtk64a/*.vtk")
unaligned_curves = []
Files = []
radii = []
Curvatures = []
Torsions = []
for idx in range(len(pre_files)):
    filename = pre_files[idx].split("\\")[-1].split(".")[0][:-8]
    if filename in ill:
        print (filename, "is found in illcases.txt, skip")
        continue
    # print (filename)
    pt, Curv, Tors, Radius, Abscissas, ptns, ftangent, fnormal, fbinormal = GetMyVtk(pre_files[idx], frenet=1)
    Files.append(pre_files[idx])
    pt = pt-np.mean(pt,axis=0)
    if SCALETO1:
        pt = pt*(1.0/measure_length(pt))
        # pt = to_unit_length(pt)
    unaligned_curves.append(pt)
    radii.append(Radius)
    Curvatures.append(Curv)
    Torsions.append(Tors)

unaligned_curves = np.array(unaligned_curves)
radii = np.array(radii)
Curvatures = np.array(Curvatures)
Torsions = np.array(Torsions)

fig = plt.figure(dpi=300,figsize=(10,6))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
window_size = 5
for i in range(len(Curvatures)):
    ax1.plot(Curvatures[i],color=cmap(0.8),alpha=0.1)
    filtered_Curvature = np.convolve(Curvatures[i], np.ones(window_size)/window_size, mode='same')
    Curvatures[i] = filtered_Curvature
    ax2.plot(filtered_Curvature,color=cmap(0.2),alpha=0.1)
    ax3.plot(Torsions[i],color=cmap(0.8),alpha=0.1)
    filtered_Torsion = np.convolve(Torsions[i], np.ones(window_size)/window_size, mode='same')
    Torsions[i] = filtered_Torsion
    ax4.plot(filtered_Torsion,color=cmap(0.2),alpha=0.1)
plt.savefig(bkup_dir+"curvature_filter.png")
plt.close()

print ("全データ（{}）を読み込みました。".format(len(pre_files)))
print ("使用できるデータ：", len(Files))

for i in range(len(Files)):
    if "BH0017_R" in Files[i]:
        base_id = i

print ("base_id:{},casename:{}で方向調整する".format(base_id, Files[base_id]))

##################################################
#  从这里开始是对齐。                             #
#  To-Do: 需要保存Procrustes对齐后的              #
#  曲线各一条，作为后续的曲线对齐的基准。           #
##################################################

a_curves = align_icp(unaligned_curves, base_id=base_id)
print ("First alignment done.")
Procrustes_curves = align_procrustes(a_curves,base_id=base_id)
print ("procrustes alignment done.")
for i in range(len(Procrustes_curves)):
    print ("length:", measure_length(Procrustes_curves[i]))
parametrized_curves = np.zeros_like(Procrustes_curves)
# aligned_curves = np.zeros_like(interpolated_curves)
for i in range(len(Procrustes_curves)):
    parametrized_curves[i] = arc_length_parametrize(Procrustes_curves[i])
Procrustes_curves = np.array(Procrustes_curves)

# if SCALETO1:
#     # 需要把长度还原到原始曲线或1
#     for i in range(len(Procrustes_curves)):
#         aligned_length = measure_length(Procrustes_curves[i])
#         procrustes_length = measure_length(Procrustes_curves[i])
#         Procrustes_curves[i] = Procrustes_curves[i] * (1.0/procrustes_length) # 这里是把长度还原到1
log.write("Scaled all curves to one.\n")


# SRVF计算
Procs_srvf_curves = np.zeros_like(Procrustes_curves)
for i in range(len(Procrustes_curves)):
    Procs_srvf_curves[i] = calculate_srvf(Procrustes_curves[i])

# Geodesic计算
log.write("- Geodesic distance is computed by SRVR, this is the only way that makes sense.\n")
Procrustes_geodesic_d = compute_geodesic_dist(Procrustes_curves)

data_item = ['Procrustes_aligned',
                'Procrustes_aligned_SRVF']
param_item = ['Curvature',
                'Torsion']
dist_item = ['Procrustes_geodesic_dist',
            'SRVF_Procrustes_geodesic_dist']
pca_standardization = 1


frechet_mean_srvf = compute_frechet_mean(Procs_srvf_curves)
frechet_mean_srvf = frechet_mean_srvf / measure_length(frechet_mean_srvf)


train_data = Procrustes_curves.reshape(len(Procrustes_curves),-1)
test_data = np.array([frechet_mean_srvf]).reshape(1,-1)
all_srvf_pca = PCAHandler(train_data, None, 16, pca_standardization)
all_srvf_pca.PCA_training_and_test()
all_srvf_pca.plot_scatter_kde(bkup_dir+"all_srvf_pca.png")






log.write("PCA standardization: {}\n".format(pca_standardization))
print ("所有PCA的标准化状态：", pca_standardization)

for loop in range(1):
    procrustes_curves = np.copy(Procrustes_curves)
    procs_srvf_curves = np.copy(Procs_srvf_curves)
    files = np.copy(Files)
    curvatures = np.copy(Curvatures)
    torsions = np.copy(Torsions)
    # 创建一个随机排列的索引
    indices = np.random.permutation(len(files))
    # 使用这个索引来重新排列 srvf_curves 和 files

    procrustes_curves = np.take(procrustes_curves, indices, axis=0)
    procs_srvf_curves = np.take(procs_srvf_curves, indices, axis=0)
    files = np.take(files, indices, axis=0)
    curvatures = np.take(curvatures, indices, axis=0)
    torsions = np.take(torsions, indices, axis=0)
    # save_shuffled_path = mkdir(bkup_dir,"shuffled_srvf_curves")
    save_shuffled_path = np.copy(bkup_dir)
    
    Scores = []

    # 获取当前时间
    now = datetime.now()

    # 将时间格式化为 'yymmddhhmmss' 格式
    formatted_time = now.strftime('%y%m%d%H%M%S')
    save_new_shuffle = mkdir(bkup_dir,formatted_time)
    loop_log = open(save_new_shuffle+"log.md", "w")

    np.save(save_new_shuffle + "file_indice.npy",files, allow_pickle=True)
    np.save(save_new_shuffle + "procrustes_curves.npy",procrustes_curves, allow_pickle=True)
    np.save(save_new_shuffle + "procs_srvf_curves.npy",procs_srvf_curves, allow_pickle=True)
    np.save(save_new_shuffle + "curvatures.npy",curvatures, allow_pickle=True)
    np.save(save_new_shuffle + "torsions.npy",torsions, allow_pickle=True)
    inverse_data_dir = mkdir(save_new_shuffle, "inverse_data")
    train_num = int(len(files)*0.75)
    test_num = int(len(files)-train_num)
    loop_log.write("# Train and test dataset split\n")
    loop_log.write("- train_num: {}\n".format(train_num))
    loop_log.write("- test_num: {}\n".format(test_num))
    train_procrustes_geodesic_d = compute_geodesic_dist(procrustes_curves[:train_num])
    test_procrustes_geodesic_d = compute_geodesic_dist(procrustes_curves[train_num:], True, np.mean(procrustes_curves[:train_num], axis=0))
    train_srvf_procrustes_geo_d = compute_geodesic_dist(procs_srvf_curves[:train_num])
    test_srvf_procrustes_geo_d = compute_geodesic_dist(procs_srvf_curves[train_num:],True, np.mean(procs_srvf_curves[:train_num], axis=0))
    train_files = files[:train_num]
    test_files = files[train_num:]

    # To-Do:Standardization
    data_dict = {
        'Procrustes_aligned': [procrustes_curves[:train_num], procrustes_curves[train_num:]],
        'Procrustes_aligned_SRVF': [procs_srvf_curves[:train_num], procs_srvf_curves[train_num:]]
    }
    param_dict = {
        'Curvature': [curvatures[:train_num], curvatures[train_num:]],
        'Torsion': [torsions[:train_num], torsions[train_num:]],
    }
    dist_dict = {
    'Procrustes_geodesic_dist': [train_procrustes_geodesic_d, test_procrustes_geodesic_d],
    'SRVF_Procrustes_geodesic_dist': [train_srvf_procrustes_geo_d, test_srvf_procrustes_geo_d]
    }
    

    loop_log.write("***\n")
    ###############################################
    # PCA
    loop_log.write("np.corrcoef is Pearson Correlation Coefficient which measures linear correlation.\n")
    loop_log.write("# coord PCA\n")
    coord_PCAs = []
    for data_key, data_values in data_dict.items():
        loop_log.write("## "+data_key+"\n")
        loop_log.write("- PCA_training_and_test will standardize data automatically.")
        train_data, test_data = data_values  # 取出列表中的两个值
        train_data=train_data.reshape(train_num,-1)
        test_data=test_data.reshape(test_num,-1)
        coord_PCAs.append(PCAHandler(train_data, test_data,standardization=pca_standardization))
        coord_PCAs[-1].PCA_training_and_test()
        components_figname = save_new_shuffle+"coord_componentse_{}.png".format(data_key)
        coord_PCAs[-1].visualize_results(components_figname)
        loading_figname = "coord_{}.png".format(data_key)
        loop_log.write("![coord_{}]({})\n".format(data_key,"./"+loading_figname))
        loop_log.write("![coord_{}]({})\n".format(data_key,"./"+components_figname))
        coord_PCAs[-1].visualize_loadings(dist_dict = dist_dict, save_path=save_new_shuffle+loading_figname)
        train_inverse = coord_PCAs[-1].inverse_transform_from_loadings(coord_PCAs[-1].train_res).reshape(train_num, -1, 3)
        test_inverse = coord_PCAs[-1].inverse_transform_from_loadings(coord_PCAs[-1].test_res).reshape(test_num, -1, 3)
        process_data_key(data_key, train_inverse, test_inverse, train_files, test_files, inverse_data_dir)
        coord_PCAs[-1].plot_scatter_kde(save_new_shuffle+"coord_scatter_kde_{}.png".format(data_key))
        coord_PCAs[-1].compute_kde()
        print ("Coords,", data_key, "JS divergence:", coord_PCAs[-1].compute_train_test_js_divergence())
        for dist_key, dist_values in dist_dict.items():
            Scores.append(ScoreHandler(data_name="Coords"+data_key, dist_name=dist_key, dist=dist_values[0], pca_result=coord_PCAs[-1].train_res, train=1))
            Scores.append(ScoreHandler(data_name="Coords"+data_key, dist_name=dist_key, dist=dist_values[1], pca_result=coord_PCAs[-1].test_res, train=0))
            
    loop_log.write("***\n")

    ###############################################
    # separate x, y, z coordinates
    loop_log.write("***\n")
    loop_log.write("# United PCA\n")
    # xyz =["x", "y", "z"]
    united_PCAs = []
    for data_key, data_values in data_dict.items():
        loop_log.write("## "+data_key+"\n")
        train_data, test_data = data_values  # 取出列表中的两个值
        train_res = []
        test_res = []
        united_internal_PCAs = []
        for i in range(3):
            united_internal_PCAs.append(PCAHandler(train_data[:,:,i], test_data[:,:,i],standardization=pca_standardization))
            united_internal_PCAs[-1].PCA_training_and_test()
            train_res_temp, test_res_temp = united_internal_PCAs[-1].train_res, united_internal_PCAs[-1].test_res
            train_res.append(train_res_temp)
            test_res.append(test_res_temp)
        train_data = np.array(train_res).transpose(1,0,2).reshape(train_num, -1)
        test_data = np.array(test_res).transpose(1,0,2).reshape(test_num, -1)
        united_PCAs.append(PCAHandler(train_data, test_data, standardization=pca_standardization))
        united_PCAs[-1].PCA_training_and_test()
        components_figname = save_new_shuffle+"united_componentse_{}.png".format(data_key)
        united_PCAs[-1].visualize_results(components_figname)
        loading_figname = "united_{}.png".format(data_key)
        loop_log.write("![united_{}]({})\n".format(data_key,"./"+loading_figname))
        loop_log.write("![united_{}]({})\n".format(data_key,"./"+components_figname))
        united_PCAs[-1].visualize_loadings(dist_dict = dist_dict, save_path=save_new_shuffle+loading_figname)
        train_inverse_large = united_PCAs[-1].inverse_transform_from_loadings(united_PCAs[-1].train_res)
        test_inverse_large = united_PCAs[-1].inverse_transform_from_loadings(united_PCAs[-1].test_res)
        train_inverse_parts = np.split(train_inverse_large, 3, axis=1)  # 分解为三个部分
        test_inverse_parts = np.split(test_inverse_large, 3, axis=1)  # 分解为三个部分
        train_inverse = []
        test_inverse = []
        for i in range(3):
            train_inverse_temp = united_internal_PCAs[i].inverse_transform_from_loadings(train_inverse_parts[i])
            test_inverse_temp = united_internal_PCAs[i].inverse_transform_from_loadings(test_inverse_parts[i])
            train_inverse.append(train_inverse_temp)
            test_inverse.append(test_inverse_temp)
        train_data_inverse = np.array(train_inverse).transpose(1,2,0)
        test_data_inverse = np.array(test_inverse).transpose(1,2,0)
        train_data_inverse = recovered_curves(train_data_inverse,"SRVF" in data_key)
        inverse_dir = mkdir(inverse_data_dir, "united_"+data_key)
        write_curves_to_vtk(train_data_inverse, train_files, inverse_dir+"train_inverse_{}.vtk".format(data_key))
        write_curves_to_vtk(test_data_inverse, test_files, inverse_dir+"test_inverse_{}.vtk".format(data_key))
        united_PCAs[-1].plot_scatter_kde(save_new_shuffle+"united_scatter_kde_{}.png".format(data_key))
        united_PCAs[-1].compute_kde()
        print ("United,",data_key, "JS divergence:", united_PCAs[-1].compute_train_test_js_divergence())
        for dist_key, dist_values in dist_dict.items():
            Scores.append(ScoreHandler(data_name="United_"+data_key, dist_name=dist_key, dist=dist_values[0], pca_result=united_PCAs[-1].train_res, train=1))
            Scores.append(ScoreHandler(data_name="United_"+data_key, dist_name=dist_key, dist=dist_values[1], pca_result=united_PCAs[-1].test_res, train=0))

    ###############################################
    loop_log.write("# Geometric param PCA\n")
    param_PCAs = []
    for data_key, data_values in param_dict.items():
        loop_log.write("## "+data_key+"\n")
        train_data, test_data = data_values  # 取出列表中的两个值
        train_data = train_data + np.random.normal(0, smooth_scale, train_data.shape)
        test_data = test_data + np.random.normal(0, smooth_scale, test_data.shape)
        loop_log.write("- PCA_training_and_test will standardize data automatically.\n")
        loop_log.write("- PCA_training_and_test will add a small amount of noise to the data.\n")
        param_PCAs.append(PCAHandler(train_data, test_data,standardization=pca_standardization))
        param_PCAs[-1].PCA_training_and_test()
        components_figname = save_new_shuffle+"param_componentse_{}.png".format(data_key)
        param_PCAs[-1].visualize_results(components_figname)
        loading_figname = "param_{}.png".format(data_key)
        loop_log.write("![param_{}]({})\n".format(data_key,"./"+loading_figname))
        loop_log.write("![param_{}]({})\n".format(data_key,"./"+components_figname))
        param_PCAs[-1].visualize_loadings(dist_dict = dist_dict, save_path=save_new_shuffle+loading_figname)
        param_PCAs[-1].plot_scatter_kde(save_new_shuffle+"param_scatter_kde_{}.png".format(data_key))
        param_PCAs[-1].compute_kde()
        print ("Param,",data_key, "JS divergence:", param_PCAs[-1].compute_train_test_js_divergence())
        for dist_key, dist_values in dist_dict.items():
            Scores.append(ScoreHandler(data_name="Param_"+data_key, dist_name=dist_key, dist=dist_values[0], pca_result=param_PCAs[-1].train_res, train=1))
            Scores.append(ScoreHandler(data_name="Param_"+data_key, dist_name=dist_key, dist=dist_values[1], pca_result=param_PCAs[-1].test_res, train=0))
    loop_log.write("***\n")

    ###############################################
    loop_log.write("***\n")
    loop_log.close()

    score_file = open(save_new_shuffle+"scores.csv", "w")
    for i in range(0, len(Scores), 2): # 2是为了区别test和train
        score_file.write("Train,")
        for score_name, score_value in Scores[i].score.items():
            score_file.write("{},".format(score_value))
        score_file.write("\n")
        score_file.write("Test,")
        for score_name, score_value in Scores[i+1].score.items():
            score_file.write("{},".format(score_value))
        score_file.write("\n")

    score_file.close()


score_files = glob.glob(bkup_dir+"shuffled_srvf_curves/*/scores.csv")
# 定义一个列表来存储每个文件的内容
all_lines = []
# 遍历score_files中的每个文件路径
for file_path in score_files:
    # 打开文件，读取每一行的内容并将其存储在列表中
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        all_lines.append(lines)

# 检查所有文件的行数是否相等
if len(set(len(lines) for lines in all_lines)) > 1:
    print('所有文件的行数不等，不能合并。')
    exit()

# 创建一个新的文件来存储合并后的内容
with open(bkup_dir+'merged_file.csv', 'w', encoding='utf-8') as mf:
    # 获取all_lines的转置，这样我们可以每次迭代一行，而不是一个文件
    for lines in zip(*all_lines):
        # 删除每一行末尾的换行符，然后将所有行内容合并，并在其后添加一个换行符
        merged_line = ''.join(line.rstrip('\n') for line in lines) + '\n'
        # 将合并后的行写入新的文件中
        mf.write(merged_line)

end_time = datetime.now()
total_time = end_time - start_time
print(dir_formatted_time, "is done in", total_time.seconds, "seconds.")
log.close()