# Standardization = z-score normalization
- *data from standardization.ipynb*
- 元データの平均を0に、標準偏差が1に変換する。

$
x_{new}^{i}=\frac{x^i-\mu}{\sigma}
$

1. (104,64,3)に対してzscoreをかける
2. x, y, z軸に分けてzscoreをかける
1と2で得られた結果は同じである。（検証済み）

## Result
standardization前後データの形（？）が変わらないことを示す。  
![](./output.png)

### PCA
![](./output2.png)
### Kmeans clustering
- cluster=3とする  
- 円周上に分布する傾向が見られた  
![](./kmeans_km3.png)

# rotationにより、z軸を消す
- オリジナルノード配置(centroidを(0,0,0)にするalignment済)    
![](./img/og_coord.png)
- rotation  
![](./img/rot_coord.png)
- Procrustes。y, zノードが線形に近い配置になっている  
![](./img/procrustes_coord.png)

## この処理の結果
distal, proximalが一致


## 以降、再standardizationを行い、PCAを行う
- clustering結果は変わらない、PCAにより変換した座標もx,y,z三軸のときとくらべてほぼ変わらない。  
- 図：○は（x,y,z）で訓練したPCAモデルresult,　ｘはrotation後、(x,y)だけで訓練したもの  
![](./img/tw_vs_tri.png)



## 各モードについて考察
- パラメータはすべてvmtkにより計算。
### BraVaデータの係数分布
![](./img/boxplot.png)

### 線形
- PC2以降0に固定、PC1だけ線形に変えてみる. すなわち、定数倍モード1だけ支配する形状
- 左図：bold gray line:current shape(x position), gray point line: mean shape(x position), red line: current curvature, purple line: previous curvature, yellow line: next curvature.
- 右図：km=3のときのk-means clustering結果、右図のprevious/current/next curveが対応するPC1とPC2の分布状況   
![](../synthetic_vessels/PC1/curvature/pillow_imagedraw.gif)
- PC3以降およびPC1を0に固定、PC2だけ線形に変えてみる. すなわち、定数倍モード2だけ支配する形状  
![](../synthetic_vessels/PC2/curvature/pillow_imagedraw.gif)
### 円周
![](../synthetic_vessels/circle/curvature/pillow_imagedraw.gif)

# 「あらゆる曲線」のPCAモデルを訓練して、「BraVa ICA」の係数分布を考察すべき