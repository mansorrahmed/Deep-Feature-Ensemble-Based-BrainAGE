
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
import re


dataShape = (121, 145, 121)
vbm_dir = "/home/imdad.khan/brain_age_mansoor/OpenBHB/VBM_OpenBHB"
vbm_files = glob.glob(os.path.join(vbm_dir, "*.npy"))
metadata_path = "/home/imdad.khan/brain_age_mansoor/OpenBHB/participants_openbhb.csv"


############## load the data matrix of size (m,i,j,k,c) where m is the number of samples,
#   i,j,k represent the 3D dimensions of the MRI, and c represents the channels ############        
def loadSplitRawData(src):
    X = np.load(src + "VBM_OpenBHB/VBM_OpenBHB.npy")
    Y = pd.read_csv(src + "SubInfoOpenBHB.csv")
    print("Data matrix loaded... shape: ", X.shape )


    ################## train-test slpit ###############
    ################## split:1
    X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X[:int(len(X)/2),:,:,:,:], Y.loc[:int(len(Y)/2)-1,"age"], test_size=0.2, random_state=42)
    print("1st train-test split completed... train shape: ", X_train_1.shape, "test shape: ", X_test_1.shape)
    np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Train1.npy" ,X_train_1)
    np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Test1.npy" ,X_test_1)
    del X_train_1, X_test_1
    print("Split subset 1 saved...")

    ################ split: 2 
    #### free the first half of the data matrix
    X = X[int(len(X)/2):,:,:,:,:] 
    Y = Y.loc[int(len(Y)/2):,"age"]
    X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("2st train-test split completed... train shape: ", X_train_2.shape, "test shape: ", X_test_2.shape)
    np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Train2.npy" ,X_train_2)
    np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Test2.npy" ,X_test_2)

    ############### merge the two sets
    del X

    print("Loading split subset 1...")
    X_train_1 = np.load(src + "VBM_OpenBHB/VBM_OpenBHB_Train1.npy")
    X_test_1 = np.load(src + "VBM_OpenBHB/VBM_OpenBHB_Test1.npy")

    X_train = np.concatenate((X_train_1, X_train_2))
    Y_train = np.concatenate((Y_train_1, Y_train_2))
    del X_train_1, X_train_2
    X_test  = np.concatenate((X_test_1, X_test_2))
    Y_test = np.concatenate((Y_test_1, Y_test_2))
    del X_test_1,X_test_2 
    print("Train-test merge completed... new train shape: ", X_train.shape, "test shape: ", X_test.shape)



# print("Data matrix saved successfully...")

####### takes the IDs and the path to the metadata file ########
def getOpenBHBLabels(IDs, metadata_path):
    metadata = pd.read_csv(metadata_path, sep="\t")
    if (type(IDs)!="pandas.core.frame.DataFrame"):
        IDs = pd.DataFrame({"participant_id":Ids}, dtype=float)
    merged = IDs.merge(metadata, left_on="participant_id", right_on="participant_id", how="inner")
    # return a dataframe with the IDs, Age, and Sex variables of the subjects
    return merged.loc[:,["participant_id", "age", "sex"]]


Ids = np.load(vbm_dir+"/IDs_OpenBHB.npy")
OpenBHB_Metadata = getOpenBHBLabels(Ids,metadata_path)
OpenBHB_Metadata.to_csv("/home/imdad.khan/brain_age_mansoor/OpenBHB/SubInfoOpenBHB.csv")