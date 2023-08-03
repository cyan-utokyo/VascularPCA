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




log = open("./log.txt", "w")
# 获取当前时间
start_time = datetime.now()
smooth_scale = 0.01
# 将时间格式化为 'yymmddhhmmss' 格式
dir_formatted_time = start_time.strftime('%y-%m-%d-%H-%M-%S')
log.write("Start at: {}\n".format(dir_formatted_time))
bkup_dir = mkdir("./", "save_data")
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
    unaligned_curves.append(pt-np.mean(pt,axis=0))
    radii.append(Radius)
    Curvatures.append(Curv)
    Torsions.append(Tors)

unaligned_curves = np.array(unaligned_curves)
radii = np.array(radii)
Curvatures = np.array(Curvatures)
Torsions = np.array(Torsions)
Curvature_changes = np.diff(Curvatures, axis=1)
Torsion_changes = np.diff(Torsions, axis=1)

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
#  To-Do: 需要保存Procrustes和MaxVariance对齐后的 #
#  曲线各一条，作为后续的曲线对齐的基准。           #
##################################################

a_curves = align_icp(unaligned_curves, base_id=base_id)
print ("First alignment done.")
Procrustes_curves = align_procrustes(a_curves,base_id=base_id)
print ("procrustes alignment done.")


parametrized_curves = np.zeros_like(Procrustes_curves)
# aligned_curves = np.zeros_like(interpolated_curves)

for i in range(len(Procrustes_curves)):
    parametrized_curves[i] = arc_length_parametrize(Procrustes_curves[i])

Aligned_curves = align_curve(parametrized_curves) # Max Variance Alignment
Procrustes_curves = np.array(Procrustes_curves) # Procrustes Alignment

# 需要把长度对齐到原始曲线
for i in range(len(Aligned_curves)):
    aligned_length = measure_length(Aligned_curves[i])
    procrustes_length = measure_length(Procrustes_curves[i])
    Aligned_curves[i] = Aligned_curves[i] * (procrustes_length/aligned_length)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

ax.hist([unaligned_curves.flatten(),Procrustes_curves.flatten(),Aligned_curves.flatten()], bins=50,
        label=["unaligned", "Procrustes aligned", "PCA aligned"],
        color=[cmap(0.9), cmap(0.7) ,cmap(0.2)], )

ax.grid(linestyle=":", alpha=0.4)
ax.set_xlabel("x(mm)")
ax.set_ylabel("Frequency")
plt.legend()
plt.savefig(bkup_dir+"histogram_alignment.png")
plt.close()

filename_procrustes_curve = bkup_dir+"procrustes_curve.vtk"
filename_pcaligned_curve = bkup_dir+"pcaligned_curve.vtk"
log.write("save histogram: {}histogram_alignment.png\n".format(bkup_dir))
log.write("save Procrustes curves: {}\n".format(filename_pcaligned_curve))
log.write("save PCA aligned curves: {}\n".format(bkup_dir))
makeVtkFile(bkup_dir+"procrustes_curve.vtk",Procrustes_curves[base_id], [],[])
makeVtkFile(bkup_dir+"pcaligned_curve.vtk", Aligned_curves[base_id], [],[])

# SRVF计算
Procs_srvf_curves = np.zeros_like(Procrustes_curves)
Pcalign_srvf_curves = np.zeros_like(Aligned_curves)
for i in range(len(Procrustes_curves)):
    Pcalign_srvf_curves[i] = calculate_srvf(Aligned_curves[i])
    Procs_srvf_curves[i] = calculate_srvf(Procrustes_curves[i])

# Geodesic计算
log.write("- Geodesic distance is computed by SRVR, this is the only way that makes sense.\n")
Procrustes_geodesic_d = compute_geodesic_dist(Procrustes_curves)
Aligned_geodesic_d = compute_geodesic_dist(Aligned_curves)
Procrustes_srvf_geodesic = compute_geodesic_dist(Procs_srvf_curves)
Aligned_srvf_geodesic = compute_geodesic_dist(Pcalign_srvf_curves)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.hist([Procrustes_geodesic_d, Aligned_geodesic_d, Procrustes_srvf_geodesic, Aligned_srvf_geodesic], bins=7,
        label=['Procrustes',"MaxVariance", 'Procrustes_SRVF', "MaxVariance_SRVF"],
        color=[cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.8)], )
ax.set_title("geodesic distance")
ax.grid(linestyle=":", alpha=0.4)
plt.legend()
plt.savefig(bkup_dir+"histogram_geodesic.png")
plt.close()


# # 获取数据集大小
# n = srvf_curves.shape[0]
log.write("proc geodsic VS aligned geodesic correlation: {}\n".format(np.corrcoef(Procrustes_geodesic_d, Aligned_geodesic_d)[0,1]))

Procs_warp_function_list, Procs_transformed_Q1_list, Procs_transformed_L1_list=compute_dtw(Procs_srvf_curves,
                                                                                            Procrustes_curves,
                                                                                            np.mean(Procs_srvf_curves,axis=0))
Pcalign_warp_function_list, Pcalign_transformed_Q1_list, Pcalign_transformed_L1_list=compute_dtw(Pcalign_srvf_curves,
                                                                                                Aligned_curves,
                                                                                                np.mean(Pcalign_srvf_curves,axis=0))
H=4
W=2
fig, ax = plt.subplots(H, W, figsize=(14, 6),dpi=200)
for i in range(len(Files)):
    ax[0,0].plot(Procrustes_curves[i,:,0],c=cmap((Procrustes_geodesic_d[i]-np.min(Procrustes_geodesic_d))/(np.max(Procrustes_geodesic_d)-np.min(Procrustes_geodesic_d))),linestyle=":")
    ax[1,0].plot(Aligned_curves[i,:,0],c=cmap((Aligned_geodesic_d[i]-np.min(Aligned_geodesic_d))/(np.max(Aligned_geodesic_d)-np.min(Aligned_geodesic_d))),linestyle=":")

    ax[0,1].plot(Procs_transformed_Q1_list[i,:,0],c=cmap((Procrustes_geodesic_d[i]-np.min(Procrustes_geodesic_d))/(np.max(Procrustes_geodesic_d)-np.min(Procrustes_geodesic_d))),linestyle=":")
    ax[1,1].plot(Pcalign_transformed_Q1_list[i,:,0],c=cmap((Aligned_geodesic_d[i]-np.min(Aligned_geodesic_d))/(np.max(Aligned_geodesic_d)-np.min(Aligned_geodesic_d))),linestyle=":")
    ax[2,0].plot(Procs_transformed_L1_list[i,:,0],c=cmap((Procrustes_geodesic_d[i]-np.min(Procrustes_geodesic_d))/(np.max(Procrustes_geodesic_d)-np.min(Procrustes_geodesic_d))),linestyle=":")
    ax[2,1].plot(Pcalign_transformed_L1_list[i,:,0],c=cmap((Aligned_geodesic_d[i]-np.min(Aligned_geodesic_d))/(np.max(Aligned_geodesic_d)-np.min(Aligned_geodesic_d))),linestyle=":")
    ax[3,0].plot(Procs_warp_function_list[i],c=cmap((Procrustes_geodesic_d[i]-np.min(Procrustes_geodesic_d))/(np.max(Procrustes_geodesic_d)-np.min(Procrustes_geodesic_d))))
    ax[3,1].plot(Pcalign_warp_function_list[i],c=cmap((Aligned_geodesic_d[i]-np.min(Aligned_geodesic_d))/(np.max(Aligned_geodesic_d)-np.min(Aligned_geodesic_d))))
for i in range(H*W):
    ax[i//2,i%2].set_title("({})".format(i+1))
for i in range(6):
    ax[i//2,i%2].set_xlabel("x(mm)",fontsize=5)
    ax[i//2,i%2].set_ylabel("y(mm)",fontsize=5)

plt.tight_layout()
plt.savefig(bkup_dir+"/x.png")
plt.close()

cmap = matplotlib.cm.get_cmap('turbo')


#titlesfile = open("./save_data/"+"titles.csv", "r")
#titles = titlesfile.read()
#titlesfile.close()
data_item = ['Variance_aligned',
             'Procrustes_aligned',
             'Variance_aligned_SRVF',
                'Procrustes_aligned_SRVF']
param_item = ['Curvature',
                'Torsion',
                'Curvature_change',
                'Torsion_change']

dist_item = ['Vatiance_geodesic_dist',
                'Procrustes_geodesic_dist',
                'SRVF_Variance_geodesic_dist',
                'SRVF_Procrustes_geodesic_dist']
pca_types = ["Coords", "United" ,"Params", "Warping"]
titles = []
total_score = []
#cores.write(titles)
for k in range(len(pca_types)):
    for i in range(len(data_item)):
        for j in range(len(dist_item)):
            titles.append(pca_types[k]+","+data_item[i]+","+dist_item[j]+",")



pca_standardization = 1
log.write("PCA standardization: {}\n".format(pca_standardization))
print ("所有PCA的标准化状态：", pca_standardization)

for loop in range(1):
    aligned_curves = Aligned_curves
    procrustes_curves = Procrustes_curves
    pcalign_srvf_curves = Pcalign_srvf_curves
    procs_srvf_curves = Procs_srvf_curves
    files = Files
    curvatures = Curvatures
    torsions = Torsions
    curvatures_changes = Curvature_changes
    torsions_changes = Torsion_changes
    # procrustes_geodesic_d = Procrustes_geodesic_d
    # aligned_geodesic_d = Aligned_geodesic_d
    # 创建一个随机排列的索引
    indices = np.random.permutation(len(files))
    # 使用这个索引来重新排列 srvf_curves 和 files
    aligned_curves = np.take(aligned_curves, indices, axis=0)
    procrustes_curves = np.take(procrustes_curves, indices, axis=0)
    pcalign_srvf_curves = np.take(pcalign_srvf_curves, indices, axis=0)
    procs_srvf_curves = np.take(procs_srvf_curves, indices, axis=0)
    files = np.take(files, indices, axis=0)
    curvatures = np.take(curvatures, indices, axis=0)
    torsions = np.take(torsions, indices, axis=0)
    save_shuffled_path = mkdir(bkup_dir,"shuffled_srvf_curves")
    fig = plt.figure(dpi=300,figsize=(10,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(np.mean(aligned_curves,axis=0)[:,0],color=cmap(0.2),alpha=0.1)
    ax2.plot(np.mean(procrustes_curves,axis=0)[:,0],color=cmap(0.2),alpha=0.1)
    for i in range(len(aligned_curves)):
        ax1.plot(aligned_curves[i,:,0],color=cmap(0.2),alpha=0.1)
        ax2.plot(procrustes_curves[i,:,0],color=cmap(0.2),alpha=0.1)
    plt.savefig(bkup_dir+"mean_curves_{}.png".format(loop))
    plt.close()
    
    Scores = []

    # 获取当前时间
    now = datetime.now()

    # 将时间格式化为 'yymmddhhmmss' 格式
    formatted_time = now.strftime('%y%m%d%H%M%S')
    save_new_shuffle = mkdir(save_shuffled_path,formatted_time)
    loop_log = open(save_new_shuffle+"log.md", "w")

    np.save(save_new_shuffle + "file_indice.npy",files, allow_pickle=True)
    np.save(save_new_shuffle + "pcaligned_curves.npy",aligned_curves, allow_pickle=True)
    np.save(save_new_shuffle + "procrustes_curves.npy",procrustes_curves, allow_pickle=True)
    np.save(save_new_shuffle + "pcalign_srvf_curves.npy",pcalign_srvf_curves, allow_pickle=True)
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
    train_aligned_geodesic_d = compute_geodesic_dist(aligned_curves[:train_num])
    test_procrustes_geodesic_d = compute_geodesic_dist(procrustes_curves[train_num:], True, np.mean(procrustes_curves[:train_num], axis=0))
    test_aligned_geodesic_d = compute_geodesic_dist(aligned_curves[train_num:], True, np.mean(aligned_curves[:train_num], axis=0))
    train_srvf_procrustes_geo_d = compute_geodesic_dist(procs_srvf_curves[:train_num])
    train_srvf_aligned_geo_d = compute_geodesic_dist(pcalign_srvf_curves[:train_num])
    test_srvf_procrustes_geo_d = compute_geodesic_dist(procs_srvf_curves[train_num:],True, np.mean(procs_srvf_curves[:train_num], axis=0))
    test_srvf_aligned_geo_d = compute_geodesic_dist(pcalign_srvf_curves[train_num:],True, np.mean(pcalign_srvf_curves[:train_num], axis=0))
    train_files = files[:train_num]
    test_files = files[train_num:]

    # To-Do:Standardization
    data_dict = {
        'Variance_aligned': [aligned_curves[:train_num], aligned_curves[train_num:]],
        'Procrustes_aligned': [procrustes_curves[:train_num], procrustes_curves[train_num:]],
        'Variance_aligned_SRVF': [pcalign_srvf_curves[:train_num], pcalign_srvf_curves[train_num:]],
        'Procrustes_aligned_SRVF': [procs_srvf_curves[:train_num], procs_srvf_curves[train_num:]]
    }
    param_dict = {
        'Curvature': [curvatures[:train_num], curvatures[train_num:]],
        'Torsion': [torsions[:train_num], torsions[train_num:]],
        'Curvature_change': [curvatures_changes[:train_num], curvatures_changes[train_num:]],
        'Torsion_change': [torsions_changes[:train_num], torsions_changes[train_num:]]
    }
    dist_dict = {
    'Variance_geodesic_dist': [train_aligned_geodesic_d, test_aligned_geodesic_d],
    'Procrustes_geodesic_dist': [train_procrustes_geodesic_d, test_procrustes_geodesic_d],
    'SRVF_Variance_geodesic_dist': [train_srvf_aligned_geo_d, test_srvf_aligned_geo_d],
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


# ----------

def align_procrustes(curves,base_id=0):
    # 使用第一条曲线作为基准曲线
    base_curve = curves[base_id]
    new_curves = []
    
    for i in range(len(curves)):
        curve = curves[i]
        # print (orthogonal(base_curve, curve, translate=True, scale=True))
        # 对齐当前曲线到基准曲线
        result = orthogonal(base_curve, curve, translate=True, scale=False)
    
        # Z是Procrustes变换后的曲线，用它来更新原始曲线
        # curves[i] = Z
        new_curves.append(result['new_b'])

    return np.array(new_curves)

def arc_length_parametrize(curve):
    # 计算曲线上每两个连续点之间的距离
    diffs = np.diff(curve, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))

    # 累加这些距离得到每个点的弧长
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

    # 生成新的参数值（等间距）
    num_points = len(curve)
    new_params = np.linspace(0, arc_lengths[-1], num_points)

    # 对曲线的每一个维度进行插值
    new_curve = np.zeros_like(curve)
    for i in range(3):
        interp = interp1d(arc_lengths, curve[:, i], kind='cubic')
        new_curve[:, i] = interp(new_params)

    return new_curve

def calculate_srvf(curve):
    # 计算曲线的导数
    t = np.linspace(0, 1, curve.shape[0])
    cs = CubicSpline(t, curve)
    derivative = cs(t, 1)
    
    # 计算SRVF
    magnitude = np.linalg.norm(derivative, axis=1)
    eps = 1e-8  # add a small positive number to avoid division by zero
    srvf = np.sqrt(magnitude + eps)[:, np.newaxis] * derivative / (magnitude[:, np.newaxis] + eps)
    return srvf

def inverse_srvf(srvf, initial_point):
    # print ("calling inverse_srvf, magnitude = np.linalg.norm(srvf, axis=1)")
    # 计算速度
    # magnitude = np.linalg.norm(srvf, axis=1)
    magnitude = np.linalg.norm(srvf, axis=1) ** 2
    velocity = srvf / np.sqrt(magnitude[:, np.newaxis])

    # 对速度进行积分
    curve = np.cumsum(velocity, axis=0)
    
    # 将曲线的初始点设为给定的初始点
    curve = curve - curve[0] + initial_point

    return curve

def align_curve(curve):
    # 将曲线平移到原点
    # print ("debug align_curve")
    # print ("curve.shape:", curve.shape)
    curve_centered = curve - np.mean(curve, axis=0)

    # Reshape the centered curve into 2D array
    curve_centered_2d = curve_centered.reshape(-1, curve_centered.shape[1]*curve_centered.shape[2])

    # 计算PCA
    pca = PCA(n_components=50)
    curve_pca = pca.fit_transform(curve_centered_2d)
    curve_pca = pca.inverse_transform(curve_pca)
    # print ("curve_pca.shape:", curve_pca.shape)

    # reshape back to the original shape
    curve_pca = curve_pca.reshape(curve_centered.shape)

    return curve_pca

def align_icp(curves,base_id=80,external_curve=None):
    # 使用第一条曲线作为基准曲线
    new_curves = []
    if base_id > -1:
        base_curve = curves[base_id]
    elif base_id == -1:
        base_curve = np.mean(curves, axis=0)
    elif base_id == -2:
        base_curve = external_curve
    
    for i in range(len(curves)):
        curve = curves[i]
        
        # 计算基准曲线和当前曲线的质心
        base_centroid = np.mean(base_curve, axis=0)
        curve_centroid = np.mean(curve, axis=0)
        
        # 将两个曲线移到原点
        base_centered = base_curve - base_centroid
        curve_centered = curve - curve_centroid
        
        # 计算最优旋转
        H = np.dot(base_centered.T, curve_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
           Vt[-1,:] *= -1
           R = np.dot(Vt.T, U.T)
        
        # 旋转和平移当前曲线以对齐到基准曲线
        # curves[i] = np.dot((curve - curve_centroid), R.T) + base_centroid
        new_curves.append(np.dot((curve - curve_centroid), R.T) + base_centroid)

    return np.array(new_curves)

def compute_geodesic_dist(curves, external=False, external_curve=None):
    geodesic_d = []
    if external == False:
        curve_A = np.mean(curves, axis=0)
    else:
        curve_A = external_curve
    for idx in range(len(curves)):
        curve_B = curves[idx]
        # 创建离散曲线对象
        discrete_curves_space = dc.DiscreteCurves(ambient_manifold=dc.Euclidean(dim=3))
        # 计算测地线距离
        geodesic_distance = discrete_curves_space.metric.dist(curve_A, curve_B)
        geodesic_d.append(geodesic_distance)
    return np.array(geodesic_d)

def recovered_curves(inverse_data, is_srvf):
    recovered_curves = []
    if is_srvf:
        for cv in inverse_data:
            recovered_curves.append(inverse_srvf(cv, np.zeros(3)))
    else:
        recovered_curves = inverse_data
    return np.array(recovered_curves)

def write_curves_to_vtk(curves, files, dir):
    for i in range(len(curves)):
        filename = files[i].split("\\")[-1].split(".")[0]
        makeVtkFile(dir+"{}.vtk".format(filename), curves[i], [],[])

def process_data_key(data_key, train_inverse, test_inverse, train_files, test_files, inverse_data_dir):
    is_srvf = "SRVF" in data_key
    train_inverse = recovered_curves(train_inverse, is_srvf)
    test_inverse = recovered_curves(test_inverse, is_srvf)

    inverse_dir = mkdir(inverse_data_dir, "coords_" + data_key)
    write_curves_to_vtk(train_inverse, train_files, inverse_dir + "train_")
    write_curves_to_vtk(test_inverse, test_files, inverse_dir + "test_")

    # print (train_inverse.shape, test_inverse.shape)



