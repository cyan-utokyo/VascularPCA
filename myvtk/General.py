import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import platform

def mkdir(super_path,testname):
    #dir_path = test_dir_path+"{}\\".format(testname)
    dir_path = super_path+"{}/".format(testname)
    if os.path.exists(dir_path)==False:
        #print("making new directory {}...".format(dir_path))
        os.mkdir(dir_path)
    # else:
    #     print("generating in directory {}...".format(dir_path))
    return dir_path


def open_folder_in_explorer(path):
    if platform.system() == "Windows":
        print ("we are in windows")
        os.system(f'explorer "{os.path.normpath(path)}"')
    elif platform.system() == "Darwin":
        print ("we are in mac")
        os.system(f'open "{path}"')
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{path}"')
    else:
        print(f"{platform.system()} is not supported")