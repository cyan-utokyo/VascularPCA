import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mkdir(super_path,testname):
    #dir_path = test_dir_path+"{}\\".format(testname)
    dir_path = super_path+"{}/".format(testname)
    if os.path.exists(dir_path)==False:
        #print("making new directory {}...".format(dir_path))
        os.mkdir(dir_path)
    # else:
    #     print("generating in directory {}...".format(dir_path))
    return dir_path

