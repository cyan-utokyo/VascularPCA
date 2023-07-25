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




log = open("./log.txt", "w")
# 获取当前时间
now = datetime.now()

# 将时间格式化为 'yymmddhhmmss' 格式
dir_formatted_time = now.strftime('%y-%m-%d-%H-%M-%S')
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
    unaligned_curves.append(pt)
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

a_curves = align_icp(unaligned_curves, base_id=base_id)
print ("First alignment done.")
Procrustes_curves = align_procrustes(a_curves,base_id=base_id)
print ("procrustes alignment done.")

# interpolated_curves = []
# for i in range(len(curves)):
#     curve = curves[i]
#     # max_pt_length = 64*4
#     # x = np.linspace(0, len(curve), len(curve))
#     # z2 = np.linspace(0,len(curve), max_pt_length)
#     # c_interpolation = interpolate.CubicSpline(x, curve)
#     # curve2 = c_interpolation(z2)
#     # interpolated_curves.append(curve2)
#     interpolated_curves.append(curve)

parametrized_curves = np.zeros_like(Procrustes_curves)
# aligned_curves = np.zeros_like(interpolated_curves)

for i in range(len(Procrustes_curves)):
    parametrized_curves[i] = arc_length_parametrize(Procrustes_curves[i])

Aligned_curves = align_curve(parametrized_curves)
Procrustes_curves = np.array(Procrustes_curves)

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


# scores.write("score,MaxVarianceAligned_Coords, ProcrustesAligned_Coords, MaxVarianceAligned_SRVF, ProcrustesAligned_SRVF, MaxVarianceAligned_United_Coords, ProcrustesAligned_United_Coords,MaxVarianceAligned_United_SRVF, ProcrustesAligned_United_SRVF, Curvature, MaxVarianceAligned_Warp_Coords, ProcrustesAligned_Warp_Coords,MaxVarianceAligned_Warp_SRVF, ProcrustesAligned_Warp_SRVF\n")

titlesfile = open("./save_data/"+"titles.csv", "r")
titles = titlesfile.read()
titlesfile.close()
scores = open(bkup_dir+"scores.csv", "w")
scores.write(titles)
scores.write("\n")



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

    Score = []
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
    # procrustes_geodesic_d = np.take(procrustes_geodesic_d, indices, axis=0)
    # aligned_geodesic_d = np.take(aligned_geodesic_d, indices, axis=0)
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
    # np.save(save_new_shuffle + "procrustes_geodesic_d.npy",procrustes_geodesic_d, allow_pickle=True)
    # np.save(save_new_shuffle + "aligned_geodesic_d.npy",aligned_geodesic_d, allow_pickle=True)

    train_num = int(len(files)*0.75)
    test_num = int(len(files)-train_num)
    loop_log.write("# Train and test dataset split\n")
    loop_log.write("- train_num: {}\n".format(train_num))
    loop_log.write("- test_num: {}\n".format(test_num))
    train_procrustes_geodesic_d = compute_geodesic_dist(procrustes_curves[:train_num])
    train_aligned_geodesic_d = compute_geodesic_dist(aligned_curves[:train_num])
    test_procrustes_geodesic_d = compute_geodesic_dist(procrustes_curves[train_num:])
    test_aligned_geodesic_d = compute_geodesic_dist(aligned_curves[train_num:])
    train_srvf_procrustes_geo_d = compute_geodesic_dist(procs_srvf_curves[:train_num])
    train_srvf_aligned_geo_d = compute_geodesic_dist(pcalign_srvf_curves[:train_num])
    test_srvf_procrustes_geo_d = compute_geodesic_dist(procs_srvf_curves[train_num:])
    test_srvf_aligned_geo_d = compute_geodesic_dist(pcalign_srvf_curves[train_num:])

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
    # for i in range(len(train_datas)):
    for data_key, data_values in data_dict.items():
        loop_log.write("## "+data_key+"\n")

        train_data, test_data = data_values  # 取出列表中的两个值
        train_data=train_data.reshape(train_num,-1)
        test_data=test_data.reshape(test_num,-1)
        train_res, test_res,pca = PCA_training_and_test(train_data, test_data, 16)
        savepath = save_new_shuffle+"coord_variance_{}.png".format(data_key)
        plot_variance(pca, train_res, test_res, savepath)
        # plot_components(components,5, save_new_shuffle+"united_components_{}.png".format(data_key))
        # for g in range(len(train_dists)):
        fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
        figname = "coord_{}.png".format(data_key)
        loop_log.write("![coord_{}]({})\n".format(data_key,"./"+figname))
        fig_count = 0
        for dist_key, dist_values in dist_dict.items():
            train_dist, test_dist = dist_values  # 取出列表中的两个值
            sc = ax[fig_count//2,fig_count%2].scatter(train_res[:,0],train_res[:,1],
                                                    c=train_dist,
                                                    cmap="turbo",
                                                    alpha=0.99,
                                                    marker="$T$")
            ax[fig_count//2,fig_count%2].scatter(test_res[:,0],test_res[:,1],
                                                c=test_dist,
                                                cmap="turbo",
                                                alpha=0.99,
                                                marker="$V$")
            # 设置子图标题的字体大小
            ax[fig_count//2,fig_count%2].set_title(dist_key, fontsize=14)
            ax[fig_count//2,fig_count%2].grid(linestyle=":", alpha=0.4)
            ax[fig_count//2,fig_count%2].set_xlabel("PC1")
            ax[fig_count//2,fig_count%2].set_ylabel("PC2")
            fig_count += 1

            train_scores, test_scores =  get_train_test_score(train_res, test_res, train_dist, test_dist)
            loop_log.write("- train scores:\n")
            for key in train_scores:
                loop_log.write("    - {}: {}\n".format(key, train_scores[key]))
                Score.append(train_scores[key])
            loop_log.write("- test scores:\n")
            for key in test_scores:
                loop_log.write("    - {}: {}\n".format(key, test_scores[key]))
                Score.append(test_scores[key])
        # 设置每个subplot的边距

        fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.85)  # 修改边距
        # 在figure的右侧添加一个colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        # 设置colorbar的标签的字体大小
        cbar.set_label('Geodesic distance', fontsize=14)
        plt.savefig(save_new_shuffle+figname)
        plt.close()

    loop_log.write("***\n")

    ###############################################
    # separate x, y, z coordinates
    loop_log.write("***\n")
    loop_log.write("# United PCA\n")
    # xyz =["x", "y", "z"]
    for data_key, data_values in data_dict.items():
        loop_log.write("## "+data_key+"\n")

        train_data, test_data = data_values  # 取出列表中的两个值
        train_res = []
        test_res = []
        for i in range(3):
            # train_res, test_res = PCA_training_and_test(train_data[:,:,i], test_data[:,:,i], 16)
            train_res_temp, test_res_temp, pca = PCA_training_and_test(train_data[:,:,i], test_data[:,:,i], 16)
            train_res.append(train_res_temp)
            test_res.append(test_res_temp)
            # loop_log.write("## {}. ".format(j+1)+titles[j]+ xyz[i]+ "\n")
            # loop_log.write("![](./"+"United_PCA_{}_{}.png".format(titles[j], xyz[i])+")\n")
        train_data = np.array(train_res).transpose(1,0,2).reshape(train_num, -1)
        test_data = np.array(test_res).transpose(1,0,2).reshape(test_num, -1)
        train_res, test_res, pca = PCA_training_and_test(train_data, test_data, 16)
        savepath = save_new_shuffle+"united_variance_{}.png".format(data_key)
        plot_variance(pca, train_res, test_res, savepath)
        # plot_components(components,5, save_new_shuffle+"united_components_{}.png".format(data_key))
        # print (train_res.shape, test_res.shape)
        fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
        figname = "United_{}.png".format(data_key)
        loop_log.write("![United_{}]({})\n".format(data_key,"./"+figname))
        fig_count = 0
        for dist_key, dist_values in dist_dict.items():
            train_dist, test_dist = dist_values  # 取出列表中的两个值
            sc = ax[fig_count//2,fig_count%2].scatter(train_res[:,0],train_res[:,1],
                                                    c=train_dist,
                                                    cmap="turbo",
                                                    alpha=0.99,
                                                    marker="$T$")
            ax[fig_count//2,fig_count%2].scatter(test_res[:,0],test_res[:,1],
                                                c=test_dist,
                                                cmap="turbo",
                                                alpha=0.99,
                                                marker="$V$")
            # 设置子图标题的字体大小
            ax[fig_count//2,fig_count%2].set_title(dist_key, fontsize=14)
            ax[fig_count//2,fig_count%2].grid(linestyle=":", alpha=0.4)
            ax[fig_count//2,fig_count%2].set_xlabel("PC1")
            ax[fig_count//2,fig_count%2].set_ylabel("PC2")
            fig_count += 1
            loop_log.write("### {}\n".format(dist_key))
            train_scores, test_scores =  get_train_test_score(train_res, test_res, train_dist, test_dist)           
            loop_log.write("- train scores:\n")
            for key in train_scores:
                loop_log.write("    - {}: {}\n".format(key, train_scores[key]))
                Score.append(train_scores[key])
            loop_log.write("- test scores:\n")
            for key in test_scores:
                loop_log.write("    - {}: {}\n".format(key, test_scores[key]))
                Score.append(test_scores[key])
        
        fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.85)  # 修改边距
        # 在figure的右侧添加一个colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        # 设置colorbar的标签的字体大小
        cbar.set_label('Geodesic distance', fontsize=14)
        plt.savefig(save_new_shuffle+figname)
        plt.close()

    ###############################################
    loop_log.write("# Geometric param PCA\n")

    for data_key, data_values in param_dict.items():
        loop_log.write("## "+data_key+"\n")
        train_data, test_data = data_values  # 取出列表中的两个值
        train_res, test_res, pca = PCA_training_and_test(train_data, test_data, 16, standardization=1)
        savepath = save_new_shuffle+"param_variance_{}.png".format(data_key)
        plot_variance(pca, train_res, test_res, savepath)
        fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
        figname = "Param_{}.png".format(data_key)
        loop_log.write("![Param_{}]({})\n".format(data_key,"./"+figname))
        fig_count = 0
        for dist_key, dist_values in dist_dict.items():
            train_dist, test_dist = dist_values  # 取出列表中的两个值
            loop_log.write("### {}\n".format(dist_key))
            train_scores, test_scores =  get_train_test_score(train_res, test_res, train_dist, test_dist)
            sc = ax[fig_count//2,fig_count%2].scatter(train_res[:,0],train_res[:,1],
                                                    c=train_dist,
                                                    cmap="turbo",
                                                    alpha=0.99,
                                                    marker="$T$")
            ax[fig_count//2,fig_count%2].scatter(test_res[:,0],test_res[:,1],
                                                c=test_dist,
                                                cmap="turbo",
                                                alpha=0.99,
                                                marker="$V$")
            # 设置子图标题的字体大小
            ax[fig_count//2,fig_count%2].set_title(dist_key, fontsize=14)
            ax[fig_count//2,fig_count%2].grid(linestyle=":", alpha=0.4)
            ax[fig_count//2,fig_count%2].set_xlabel("PC1")
            ax[fig_count//2,fig_count%2].set_ylabel("PC2")
            fig_count += 1
            loop_log.write("- train scores:\n")
            for key in train_scores:
                loop_log.write("    - {}: {}\n".format(key, train_scores[key]))
                Score.append(train_scores[key])
            loop_log.write("- test scores:\n")
            for key in test_scores:
                loop_log.write("    - {}: {}\n".format(key, test_scores[key]))
                Score.append(test_scores[key])
        fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.85)  # 修改边距
        # 在figure的右侧添加一个colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        # 设置colorbar的标签的字体大小
        cbar.set_label('Geodesic distance', fontsize=14)
        plt.savefig(save_new_shuffle+figname)
        plt.close()
    loop_log.write("***\n")

    ###############################################
    loop_log.write("# Warping function PCA\n")
    for data_key, data_values in data_dict.items():

        loop_log.write("## "+data_key+"\n")
        train_data_pre, test_data_pre = data_values  # 取出列表中的两个值
        train_warping_functions= []
        mean_shape = np.mean(train_data_pre, axis=0)
        for j in range(len(train_data_pre)):
            distance, path = calculate_dtw(train_data_pre[j], mean_shape)
            train_warping_functions.append(get_warping_function(path, train_data_pre[j], mean_shape))
        test_warping_functions = []
        for j in range(len(test_data_pre)):
            distance, path = calculate_dtw(test_data_pre[j], mean_shape)
            test_warping_functions.append(get_warping_function(path, test_data_pre[j], mean_shape))

        train_data = np.array(train_warping_functions)
        test_data = np.array(test_warping_functions)
        train_res, test_res, pca = PCA_training_and_test(train_data, test_data, 16, standardization=1)
        savepath = save_new_shuffle+"warping_variance_{}.png".format(data_key)
        plot_variance(pca, train_res, test_res, savepath)
        # # plot_components(components, 5, save_new_shuffle+"warping_components_{}.png".format(data_key))
        fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
        figname = "Warp_{}.png".format(data_key)
        loop_log.write("![Warp_{}]({})\n".format(data_key,"./"+figname))
        fig_count = 0
        for dist_key, dist_values in dist_dict.items():
            train_dist, test_dist = dist_values  # 取出列表中的两个值
            loop_log.write("### {}\n".format(dist_key))
            sc = ax[fig_count//2,fig_count%2].scatter(train_res[:,0],train_res[:,1],
                                                    c=train_dist,
                                                    cmap="turbo",
                                                    alpha=0.99,
                                                    marker="$T$")
            ax[fig_count//2,fig_count%2].scatter(test_res[:,0],test_res[:,1],
                                                c=test_dist,
                                                cmap="turbo",
                                                alpha=0.99,
                                                marker="$V$")
            # 设置子图标题的字体大小
            ax[fig_count//2,fig_count%2].set_title(dist_key, fontsize=14)
            ax[fig_count//2,fig_count%2].grid(linestyle=":", alpha=0.4)
            ax[fig_count//2,fig_count%2].set_xlabel("PC1")
            ax[fig_count//2,fig_count%2].set_ylabel("PC2")
            fig_count += 1
            loop_log.write("- train scores:\n")
            for key in train_scores:
                loop_log.write("    - {}: {}\n".format(key, train_scores[key]))
                Score.append(train_scores[key])
            loop_log.write("- test scores:\n")
            for key in test_scores:
                loop_log.write("    - {}: {}\n".format(key, test_scores[key]))
                Score.append(test_scores[key])
        fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.85)  # 修改边距
        # 在figure的右侧添加一个colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        # 设置colorbar的标签的字体大小
        cbar.set_label('Geodesic distance', fontsize=14)
        plt.savefig(save_new_shuffle+figname)
        plt.close()
    loop_log.write("***\n")
    
    for i in range(len(Score)):
        scores.write(str(Score[i])+",")
    scores.write("\n")




    loop_log.close()

log.close()
scores.close()
print (dir_formatted_time, "done")