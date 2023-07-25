import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mkdir(super_path,testname):
    #dir_path = test_dir_path+"{}\\".format(testname)
    dir_path = super_path+"{}/".format(testname)
    if os.path.exists(dir_path)==False:
        print("making new directory {}...".format(dir_path))
        os.mkdir(dir_path)
    else:
        print("generating in directory {}...".format(dir_path))
    return dir_path

def get_scores(res, dist):
    scores = {}
    train_pca_dist = []
    for j in range(res.shape[0]):
        train_pca_dist.append(np.linalg.norm(res[j]))
    correlation_matrix = np.corrcoef(train_pca_dist, dist) #dists[i])
    scores["correlation_coef"] = correlation_matrix[0,1]
    train_cosine_similarity = cosine_similarity([train_pca_dist, dist])[0,1]
    scores["cosine_similarity"] = train_cosine_similarity
    return scores

def get_train_test_score(train_res, test_res, train_dist, test_dist):
    train_scores = get_scores(train_res, train_dist)
    test_scores = get_scores(test_res, test_dist)
    return train_scores, test_scores