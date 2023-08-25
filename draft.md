# intro

# Method
## Geometric param of CMA?
1. Curvature & Torsion是如何计算出来的
2. CMA的Geom param有什么特征
3. **理论上**,CUSV应当有什么特征（以及利用这些特征开发出的自动分类算法的简要介绍）

# 曲线的表达形式
1. 为什么选择曲线的表达形式是重要的？（几乎相当于*为什么需要用SRVF表达一条曲线*）因为只有正确的表达形式的参数间差异才能反映实际形状的差异。
    - 举个例子：在坏的参数化形式下，同一形状可能有不同表达形式，不同形式之间有计算上的差异，这导致明明是同一个形状，却在计算上不同
    - 为什么上述现象是不好的？（这段纯粹是写给玛丽看）因为我们希望达成的效果是PCA空间中的一点对应唯一的一个形状
2. 引入SRVF：这是一种用标准化速度表达曲线形状的方式。它的式子是~，因此它不受影响。
3. 在引入SRVF的基础上介绍Geodesic，但具体要在**metric**一节说明。

## 坐标的PCA
1. 什么是PCA：一种对多维特征的正交操作，可以得到一组互不相关的特征
2. 坐标的PCA：如何处理才能把坐标视为一组特征？
    - 必须保证它们的对应位置的landmark指向同一个解剖学上的位置，因此需要对齐和按照弧长重参数化，最后标准化数据
    - 介绍什么是procrustes对齐
    - 介绍什么是按照弧长重新参数化
    - 使用什么标准化手段，达成了什么效果（每一组对应特征的均值为0，方差为1）
3. PCA分析能得到一组主成分，如何使用这组主成分？
    1. 可以拆分一条形状为一组主成分（式子），也可以利用`PC loadings` 还原一条形状
    2. 这些PC loadings即是原来的曲线在PC空间中的映射，也即可以把这条高维曲线用低维特征表达出来
    3. *KDE似乎要插在这里*

## Metric：Geodesic距离
- 是一个被构筑出来的manifold上的L2距离，构筑该manifold的metric是fisher-rao距离。也就是说，在这个manifold上，任意两点的距离是fisher-rao距离（？？？）
- 它是一个标量，可以用来有效衡量两条曲线的形状的差异。


# Result
## CMA的抽取和CUSV的分类
- 使用BraVa中的87根血管
- 使用算法分类，然后由具有相关背景的学生校验并修正算法
### 结果
- U:42(49%)
- V:29(33%)
- C:9(11%)
- S:6(7%)
## Geom param of CMA
认为各type曲率的均值是典型的该type曲率
### 各类别平均$\kappa$和$\tau$的特征（直观）
- To-do: 这部分还没引入对齐和重新采样，所以X轴必须是标准化长度
![和无造作的BraVa平均相比](./bkup_dir/23-08-11-16-05-32/geometry/Curvatures_Torsions.png)
![和排除bias的BraVa平均相比](./bkup_dir/23-08-12-14-42-23/geometry/debias_Curvatures_Torsions.png)
- To-Do: 用各类平均代表一类，求四类平均的平均来debias的方法并不好。
- ~~To-Do:需要两张图，第一章是和无造作的BraVa平均相比各type 的param偏差，另一张是和排除了偏差的BraVa平均形状相比。~~
### 各类别平均$\kappa$和$\tau$的特征（数值）
![各type平均形状与无造作的BraVa平均的偏差](./bkup_dir/23-08-11-16-05-32/geometry/group_param_compare.png)
![各type平均形状与de-bias BraVa平均的偏差]()

## 曲线的SRVF和geodesic距离
- 选取基准点：算术平均形状和frechet平均形状（？？意义不明）
- 几种对比方法
    1. 是否$\kappa$和$\tau$相近的形状的geodesic距离就更近？（To-Do: 要搞几个例子）
    2. 是否同一人的左右形状的geodesic距离就更近？

## 坐标的PCA
### 位置坐标的PCA
![位置坐标的PCA和SRVF的PCA](./bkup_dir/23-08-11-18-04-51/pca_analysis/PCA_total.png)
![位置PCA](./bkup_dir/23-08-11-18-04-51/pca_analysis/pca_plot_variance.png)
![srvfPCA](./bkup_dir/23-08-11-18-04-51/pca_analysis/srvf_pca_plot_variance.png)
### 两种PCA的PCA loading的统计分布
![位置坐标PCA的loading分布](./bkup_dir/23-08-11-16-05-32/pca_analysis/PCA_total_Violinplot.png)
![SRVF坐标PCA的loading分布](./bkup_dir/23-08-11-16-05-32/pca_analysis/srvfPCA_total_Violinplot.png)
### 使用KDE拟合各type的loading分布状况
- 它们组成一个高维点云，这个点云在2D映射下长这样，可以看出同一类别在低维（2d）上的映射是不连续的，也就是说无法某一类别内的形状并不会受单一PC影响。
- To-Do: 一个2d KDE+scatter

### To-Do: 求任意两条线之间的geodesic并给出统计