# Train and test dataset split
- train_num: 43
- test_num: 44
***
np.corrcoef is Pearson Correlation Coefficient which measures linear correlation.
# coord PCA
## Variance_aligned
- PCA_training_and_test will standardize data automatically.![coord_Variance_aligned](./coord_Variance_aligned.png)
![coord_Variance_aligned](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/coord_componentse_Variance_aligned.png)
## Procrustes_aligned
- PCA_training_and_test will standardize data automatically.![coord_Procrustes_aligned](./coord_Procrustes_aligned.png)
![coord_Procrustes_aligned](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/coord_componentse_Procrustes_aligned.png)
## Variance_aligned_SRVF
- PCA_training_and_test will standardize data automatically.![coord_Variance_aligned_SRVF](./coord_Variance_aligned_SRVF.png)
![coord_Variance_aligned_SRVF](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/coord_componentse_Variance_aligned_SRVF.png)
## Procrustes_aligned_SRVF
- PCA_training_and_test will standardize data automatically.![coord_Procrustes_aligned_SRVF](./coord_Procrustes_aligned_SRVF.png)
![coord_Procrustes_aligned_SRVF](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/coord_componentse_Procrustes_aligned_SRVF.png)
***
***
# United PCA
## Variance_aligned
![united_Variance_aligned](./united_Variance_aligned.png)
![united_Variance_aligned](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/united_componentse_Variance_aligned.png)
## Procrustes_aligned
![united_Procrustes_aligned](./united_Procrustes_aligned.png)
![united_Procrustes_aligned](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/united_componentse_Procrustes_aligned.png)
## Variance_aligned_SRVF
![united_Variance_aligned_SRVF](./united_Variance_aligned_SRVF.png)
![united_Variance_aligned_SRVF](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/united_componentse_Variance_aligned_SRVF.png)
## Procrustes_aligned_SRVF
![united_Procrustes_aligned_SRVF](./united_Procrustes_aligned_SRVF.png)
![united_Procrustes_aligned_SRVF](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/united_componentse_Procrustes_aligned_SRVF.png)
# Geometric param PCA
## Curvature
- PCA_training_and_test will standardize data automatically.
- PCA_training_and_test will add a small amount of noise to the data.
![param_Curvature](./param_Curvature.png)
![param_Curvature](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/param_componentse_Curvature.png)
## Torsion
- PCA_training_and_test will standardize data automatically.
- PCA_training_and_test will add a small amount of noise to the data.
![param_Torsion](./param_Torsion.png)
![param_Torsion](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/param_componentse_Torsion.png)
## Curvature_change
- PCA_training_and_test will standardize data automatically.
- PCA_training_and_test will add a small amount of noise to the data.
![param_Curvature_change](./param_Curvature_change.png)
![param_Curvature_change](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/param_componentse_Curvature_change.png)
## Torsion_change
- PCA_training_and_test will standardize data automatically.
- PCA_training_and_test will add a small amount of noise to the data.
![param_Torsion_change](./param_Torsion_change.png)
![param_Torsion_change](././save_data/23-08-01-21-22-09/shuffled_srvf_curves/230801212324/param_componentse_Torsion_change.png)
***
# Warping function PCA
- only use the last two data (SRVF) to compute warping function, but use another two data (Coords) to compute geodesic distance.
- Cosine similarity is used to compute warping function.
- ***
