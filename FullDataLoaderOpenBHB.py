import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import os, io, glob, re
# import nilearn as nl
# from nilearn.masking import apply_mask
# from nilearn.image import index_img
# from nilearn.input_data import NiftiMasker
from sklearn.model_selection import train_test_split
# import nibabel as nib
import gc, argparse


class DataLoaderOpenBHB:
    def __init__(self) -> None:
        pass
    
    # the filenames in the training set don't contain the exactly matching unique IDs
    # as in the metadata file, therefore last 6 digits in the ID need to be rounded to zeros
    # rename the file names (round last 6 digits)
    def renameFiles(self, dir, modality):
        # 'dir' is the directory containing the current files
        if (modality =="vbm_gm"):
            # fpaths contains the list of the all files in the 'dir'
            fpaths = glob.glob(os.path.join(dir, "*cat12vbm_desc-gm_T1w.npy"))
            str_pattern = "_preproc-cat12vbm_desc-gm_T1w"
            self.rename_mri(fpaths, str_pattern)
        elif (modality == "quasi_raw"):
            fpaths = glob.glob(os.path.join(dir, "*quasiraw_T1w.npy"))
            str_pattern = "_preproc-quasiraw_T1w.npy"
            # rename the files in the current directory according to the str-pattern specified
            self.rename_mri(fpaths, dir, str_pattern)
            print("Files renamed successfully...")


    def rename_mri(self, fpaths, dir, str_pattern):
        for idx, val in enumerate(fpaths):
            # iterate through all file paths in 'dir'
            id_i = re.search("sub-(.*)_prep", str(val))
            id_i = round(int(id_i.group(1)), -6)
            new_file = dir + "sub-" + str(id_i) + str_pattern
            old_file = val
            os.rename(old_file , new_file )

    def mri2dmatrix(self, dir, dataShape, modality, save=True):
        """
        -> This function takes raw MRIs in the src directory
        -> Outputs a (samples, channels, x, y, z) tensor/numpy array
        """
        self.modality = modality

        if (self.modality =="VBM"):
            # fpaths contains the list of the all files in the 'dir'
            fpaths = glob.glob(os.path.join(dir, "*cat12vbm_desc-gm_T1w.npy"))
        elif (self.modality == "QuasiRaw"):
            fpaths = glob.glob(os.path.join(dir, "*quasiraw_T1w.npy"))

        X = np.empty((len(fpaths),1, dataShape[0],dataShape[1],dataShape[2]))
        Ids = []

        for i in range(len(fpaths)):
            id_i = re.search("sub-(.*)_prep", str(fpaths[i])).group(1)
            X[i,:,:,:,:] = np.load(fpaths[i]).reshape((1,)+dataShape)
            Ids.append(id_i)
            if (i%100==0):
                print(i," files loaded... shape: ", X.shape)
        print("Data matrix created with shape: ", X.shape)
        if (save):
            np.save(dir+ self.modality + "_OpenBHB.npy", X)
            np.save(dir+ self.modality + "_IDs_OpenBHB.npy", Ids)

            print("Data matrix saved successfully...")
        return Ids


    ####### takes the IDs and the path to the metadata file ########
    def getOpenBHBLabels(self,IDs, metadata_path):
        """"
        This function takes the Subject-IDs and return a dataframe containing their age and gender.
        """
        metadata = pd.read_csv(metadata_path)
        if (type(IDs)!="pandas.core.frame.DataFrame"):
            IDs = pd.DataFrame({"participant_id":Ids}, dtype=float)
        merged = IDs.merge(metadata, left_on="participant_id", right_on="participant_id", how="inner")
        # return a dataframe with the IDs, Age, and Sex variables of the subjects
        OpenBHB_Metadata = merged.loc[:,["participant_id", "age", "sex"]]
        OpenBHB_Metadata.to_csv(metadata_path +self.modality+ "SubInfoOpenBHB.csv")
        print("Metadata file created and saved to the destination directory...")
    

    ############## load the data matrix of size (m,i,j,k,c) where m is the number of samples,
    #   i,j,k represent the 3D dimensions of the MRI, and c represents the channels ############        
    def loadSplitRawData(self, X_file, Y_file, src, modality, splitInChunks=False):
        """
        - loads X (numpy array) and Y (csv file) with columns [participant_id, age, sex]
        - creates training and test sets 
        """
        X = np.load(src + X_file + ".npy")
        Y = pd.read_csv(src + Y_file + ".csv")
        labels = Y.loc[:,"age"]
        print("Data matrix loaded... shape: X = ", X.shape, " Y = ", Y.shape )

        if (splitInChunks):
            ################## train-test slpit ###############
            ################## split:1
            X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X[:int(len(X)/2),:,:,:,:], Y.loc[:int(len(Y)/2)-1,"age"], test_size=0.2, random_state=42)
            print("1st train-test split completed... train shape: ", X_train_1.shape, "test shape: ", X_test_1.shape)
            np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Train1.npy" ,X_train_1)
            np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Test1.npy" ,X_test_1)
            del X_train_1, X_test_1
            gc.collect()

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
            gc.collect()

            print("Loading split subset 1...")
            X_train_1 = np.load(src + "VBM_OpenBHB/VBM_OpenBHB_Train1.npy")
            X_test_1 = np.load(src + "VBM_OpenBHB/VBM_OpenBHB_Test1.npy")

            X_train = np.concatenate((X_train_1, X_train_2))
            Y_train = np.concatenate((Y_train_1, Y_train_2))
            del X_train_1, X_train_2
            gc.collect()
            X_test  = np.concatenate((X_test_1, X_test_2))
            Y_test = np.concatenate((Y_test_1, Y_test_2))
            del X_test_1,X_test_2 
            gc.collect()
            print("Train-test merge completed... new train shape: ", X_train.shape, "test shape: ", X_test.shape)
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            np.save(src + "X_train_" + modality + ".npy", X_train)
            np.save(src + "X_test_"+ modality + ".npy", X_test)
            np.save(src + "Y_train_" + modality + ".npy", Y_train)
            np.save(src + "Y_test_" + modality + ".npy", Y_test)
            print("Train-test split completed and saved...\n X-train ", X_train.shape,
                   "X-test ", X_test.shape)
            
    """
    Returns the specified mask (img) file 
    """
    def getMask(self, modality):
        MASKS = {
            "vbm": {
                "basename": "cat12vbm_space-MNI152_desc-gm_TPM.nii",
                "thr": 0.05},
            "quasiraw": {
                "basename": "quasiraw_space-MNI152_desc-brain_T1w.nii",
                "thr": 0}
        }

        img = nib.load('/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/resource/cat12vbm_space-MNI152_desc-gm_TPM.nii')

        mask_vbm = np.array(img.dataobj)

        img_quasi_mask = nib.load('/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/resource/quasiraw_space-MNI152_desc-brain_T1w.nii').get_fdata()
        # im = nibabel.Nifti1Image(arr.squeeze(), affine)
        # arr = apply_mask(im, masks[key])

        # mask = np.array(img_quasi_mask.dataobj)

        # retuns a 2D array with ones on the diagonals and zeros on non-diagnonal enteries
        self.affine = np.eye(4)
        resourcedir = '/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/resource/'
        self.masks = dict((key, os.path.join(resourcedir, val["basename"]))
                        for key, val in MASKS.items())
        for key in self.masks:
            arr = nib.load(self.masks[key]).get_fdata()
            thr = MASKS[key]["thr"]
            arr[arr <= thr] = 0
            arr[arr > thr] = 1
            self.masks[key] = nib.Nifti1Image(arr.astype(int), self.affine)

        arr = index_img('/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/resource/cat12vbm_space-MNI152_desc-gm_TPM.nii',0).get_fdata()
        thr = MASKS["vbm"]["thr"] #### thr = 0.05
        arr[arr <= thr] = 0
        arr[arr > thr] = 1
        self.mask_vbm = nib.Nifti1Image(arr.astype(int), self.affine)


    """"
    This function loads all data (.npy) modalities (roi, vbm, sbm, quasi-raw)
    Flattens the array and merges all modalities together in an 2d matrix
    Saves the merged 2D matrix (.npy) locally
    """
    def loadDumpFullData(self, all_paths):
        participants = pd.DataFrame(dtype=float)
        for i in range(0,3209):
            for j in range(0,3):
                if j == 0:
                    vbm_roi_i = np.load(all_paths[i,j]).flatten().reshape(1,284)
                if j ==1:
                    deskn_roi_i = np.load(all_paths[i,j]).flatten().reshape(1,476)
                if j == 2:
                    destrx_roi_i = np.load(all_paths[i,j]).flatten().reshape(1,1036)
                if j==3:
                    # masking of the CAT12 VBM
                    unmasked_vbm_i = np.load(all_paths[i,j], mmap_mode="r")
                    img = nib.Nifti1Image(unmasked_vbm_i.squeeze(), self.affine)
                    masked_vbm_i = apply_mask(img, self.mask_vbm)
                    masked_vbm_i = np.expand_dims(masked_vbm_i, axis=0)
                if j==4:
                    fsl_xhemi_i = np.load(all_paths[i,j]).flatten().reshape(1,1310736)
                if j==5:
                    # masking of the QuasiRaw MRI
                    unmasked_quasi_i = np.load(all_paths[i,j], mmap_mode="r")
                    im = nib.Nifti1Image(unmasked_quasi_i.squeeze(), self.affine)
                    masked_quasi_i = apply_mask(im, self.masks['quasiraw'])
                    masked_quasi_i = np.expand_dims(masked_quasi_i, axis=0)
            
            # extract subject unique IDs from the file_name
            # round-off last 6 digits in the ID and insert zeros inplace 
            id_i = re.search("sub-(.*)_prep", all_paths[i,j])
            id_i = round(int(id_i.group(1)), -6)
            
            age = participants["age"].loc[participants['participant_id'] == id_i]
            gender = participants["sex"].loc[participants['participant_id'] == id_i]
            age = age.astype(float)
            # sub_metadata = sub_metadata.append([[id_i,age.values, gender.values]], ignore_index=True)
            concat_row_i = np.concatenate((np.array(id_i).reshape(1,1), vbm_roi_i, deskn_roi_i,
                    destrx_roi_i, 
                    np.array(gender.values).reshape(1,1), np.array(age.values).reshape(1,1)) , axis=1)
            # print("Total size of a full row (1,3659575): ", concat_row_i.nbytes, "bytes")
            with open ("/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/openbhb_train_roi_dataset.npy",'a') as f_object:
                writer_object = io.writer(f_object, delimiter=",")
                writer_object.writerow(concat_row_i.reshape(1799))
                f_object.close()
    # print("Data matrix saved successfully...")




if __name__ == '__main__':

    dataLoader = DataLoaderOpenBHB()

    parser = argparse.ArgumentParser(description="Create a data tensor by combining the raw MRI files",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rawsrc", type=str, default="", nargs="?", const="", help="Raw MRIs source location")
    parser.add_argument("--metadatapath",  type=str, default="", nargs="?", const="",  help="Metadata (csv) file location - "
                        "should have three mandatory columns: participant_id, age, and sex") 
    parser.add_argument("--dataShape", type=tuple, help="Shape of an MRI [default 121*145*121]", nargs='?', default=(121, 145, 121), const=(121, 145, 121))
    parser.add_argument("--modality", type=str,  default="", nargs="?", const="", help="Modality - either QuasiRaw or VBM")
    
    args = vars(parser.parse_args())
    mri_dir = args["rawsrc"]
    metadata_path = args["metadatapath"]
    dataShape = args["dataShape"]
    modality = args["modality"]

    dataShape = (182, 218, 182)
    # mri_dir = "/home/imdad.khan/brain_age_mansoor/OpenBHB/VBM_OpenBHB/"
    # metadata_path = "/home/imdad.khan/brain_age_mansoor/OpenBHB/participants_openbhb.csv"
    # rename_dir = '/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/train/train_quasiraw/'

    # dataLoader.renameFiles(rename_dir, modality=modality)

    X_file="QuasiRaw_OpenBHB"
    Y_file="QuasiRawSubInfoOpenBHB"
    src="/media/dataanalyticlab/Drive2/MANSOOR/Dataset/Test/OpenBHB/Quasi-Raw/"

    if (len(mri_dir)>0):    
        Ids = dataLoader.mri2dmatrix(dir=mri_dir, dataShape=dataShape, 
                                     modality=modality, save=True)
        dataLoader.getOpenBHBLabels(Ids,metadata_path)
        
    dataLoader.loadSplitRawData(X_file=X_file, Y_file=Y_file,
                                src=src,
                                modality=modality)
    



