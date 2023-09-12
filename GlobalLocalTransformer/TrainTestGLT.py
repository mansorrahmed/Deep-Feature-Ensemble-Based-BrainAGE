#!/usr/bin/env python

import warnings
warnings.simplefilter("ignore")

##### dependencies ###
# pytorch
# numpy
# sklearn
#####################

from GlobalLocalTransformer import *
import torch 
import torch.nn as nn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
import argparse
from argparse import ArgumentParser
import time
import warnings
warnings.filterwarnings("ignore")

class Trainer:
    """
    The simple trainer.
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
            
    # s_start = starting slice, s_end = ending slice
    def loadFullDataGLT(self, source_path, s_start, s_end,  labels_path=""):
        """
        returns the sliced tensors from the training and test set
        """
        print("Started loading data..")
        X_train = np.load(source_path + "X_train_VBM_OpenBHB.npy")
        Y_train = np.load(source_path + "Y_train_VBM_OpenBHB.npy")

        X_test = np.load(source_path + "X_test_VBM_OpenBHB.npy")
        Y_test = np.load(source_path + "Y_test_VBM_OpenBHB.npy")
        
        print("Data tensor loaded having shapes:> X-train: ", X_train.shape, 
              "X-test: ", X_test.shape, "Y-train: ", Y_train.shape, "Y-test: ", Y_test.shape )
        # labels = Y.loc[:,"age"]

        # slice the 3D MRI volume and then reshape the tensor
        X_train_s = X_train[:,:,:,s_start:s_end,0]
        X_test_s = X_test[:,:,:,s_start:s_end,0]

        del X_train, X_test

        # X_train, X_test, Y_train, Y_test = train_test_split(X_s, labels, test_size=0.2, random_state=42)

        X_train_s = torch.from_numpy(X_train_s)
        X_train_s = X_train_s.permute(0,3,1,2)

        X_test_s = torch.from_numpy(X_test_s)
        X_test_s = X_test_s.permute(0,3,1,2)
        print("Sliced training set size: ", X_train_s.shape, ", test set size: ", X_test_s.shape)

        # Y_train = torch.tensor(np.array(Y_train))
        # Y_test = torch.tensor(np.array(Y_test))
        # torch.tensor()

        # print(Y_train.shape, Y_test.shape)
        return X_train_s, X_test_s,  Y_train, Y_test
  
    def createDataLoaderGLT(self,src,numTrainSamples, numTestSamples,batchSize=32,num_workers=4):
        # Create memory-mapped arrays
        X_train = np.memmap(src + 'X_train_VBM_OpenBHB.npy', dtype=float, mode='r', shape=(numTrainSamples,121, 145, 121,1))
        Y_train = np.memmap(src + 'Y_train_VBM_OpenBHB.npy', dtype=float, mode='r', shape=(numTrainSamples,))
        X_test = np.memmap(src + 'X_test_VBM_OpenBHB.npy', dtype=float, mode='r', shape=(numTestSamples,121, 145, 121,1))
        Y_test = np.memmap(src + 'Y_test_VBM_OpenBHB.npy', dtype=float, mode='r', shape=(numTestSamples,))
        print("Memory maps of the data tensors created...")

        # Custom Dataset class using memory-mapped arrays
        class myDataset(Dataset):
            def __init__(self, X, y):
                self.data = X
                self.targets = y

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                sample = self.data[idx]
                target = self.targets[idx]
                return sample, target

        # Instantiate custom dataset
        train_dataset = myDataset(X_train, Y_train)
        test_dataset = myDataset(X_test, Y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=batchSize, num_workers=num_workers, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batchSize, num_workers=num_workers, shuffle=True)
        return train_dataloader, test_dataloader


    def trainGLT(self, model_dir, attention_model, regression_model,   criterion, optimizer, nEpochs, batchSize,
                 X_train="", X_test="", Y_train="", Y_test="", loadData=True,src="",
                 numTrainSamples=0, numTestSamples =0, num_workers=1):
        self.model_dir = model_dir
        self.attention_model = attention_model
        self.regression_model = regression_model
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = nEpochs
        self.batch_size = batchSize
        self.train_loss = [] 
        self.val_acc = []
        self.loadData = loadData

        # check if data is already loaded in the memory, otherwise split it into batches
        if (loadData):
            self.train_dataloader = DataLoader(list(zip(X_train, Y_train)), batch_size=batchSize, shuffle=True)
            self.test_dataloader = DataLoader(list(zip(X_test, Y_test)), batch_size=batchSize, shuffle=True)
        else:
            self.train_dataloader,self.test_dataloader = self.createDataLoaderGLT(src,
                                                                                  numTrainSamples,
                                                                                  numTestSamples,
                                                                                  batchSize,
                                                                                  num_workers)
        print("Created train-test data loaders...")

        # Loop through the number of epochs
        for self.epoch in range(nEpochs):
            begin = time.time()

            self.train_epoch_loss = 0  # Initialize loss for the entire epoch
            self.test_epoch_mae = 0   # Initialize the total MAE for the entire epoch
            optimizer.zero_grad()  # a clean up step for PyTorch

            loss = trainEpoch(self)
            self.train_loss.append(loss)

            
            mae = evaluateGLT(self)
            self.val_acc.append(mae)

            time.sleep(1)
            # store end time
            end = time.time()
            print("Epoch ", self.epoch, " competed in time: ", round((end - begin), 2))
            
        return self.train_loss , self.val_acc
    

def trainEpoch(self):
    """
    Training loop -> train the model using the entire training set for one complete epoch 
    """   
    print("Training epoch ",self.epoch," started.." )

    # Loop through the batches and update the gradients after each batch
    for X_batch, y_batch in self.train_dataloader:
        if (self.loadData==False):
            X_batch = X_batch[:,:,:,55:60,0]
            X_batch = X_batch.permute(0,3,1,2)
        # Obtaining the cuda parameters
        X_batch = X_batch.to(device=self.device)
        y_batch = y_batch.to(device=self.device)
        
        self.regression_model.train()
        zlist = self.attention_model(X_batch)             # forward pass
        output_tensors = torch.cat(zlist, dim=1).to(device=self.device)
        self.batch_loss = 0
        # loop through each sample in the current batch 
        for i in range(len(X_batch)):
            # get the output values (ages) as the output of the last layer
            output = self.regression_model(output_tensors[i]).to(device=self.device)
            loss = self.criterion(output, y_batch[i])  # Compute the loss for each sample
            self.batch_loss += loss  # Accumulate the loss for the entire batch
            self.train_epoch_loss += loss  # Accumulate the loss for the entire epoch
            
        batch_avg_loss = self.batch_loss / len(X_batch)  # Calculate the average loss for the batch
        batch_avg_loss.backward()  # Backward pass (compute parameter updates)
        # print("Batch completed")
        self.optimizer.step()


    train_epoch_avg_loss = round((self.train_epoch_loss / len(self.train_dataloader.dataset)).item(),3)
    print("Epoch:", self.epoch, "Loss:", train_epoch_avg_loss)
    # Save the model's state dictionary after each epoch
    model_path = f'{self.model_dir}model_epoch_{self.epoch}.pth'
    torch.save(self.model.state_dict(), model_path)

    return train_epoch_avg_loss


def evaluateGLT(self):
    """
    Validation loop -> evaluate the model on the entire test set for one epoch
    """
    print("Validation epoch ",self.epoch," started.." )
    self.regression_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # loop through each batch of the test set and compute the evaluation metrics
        for X_batch, y_batch in self.test_dataloader:
            if (self.loadData==False):
                X_batch = X_batch[:,:,:,55:60,0]
                X_batch = X_batch.permute(0,3,1,2)

            # Obtaining the cuda parameters
            X_batch = X_batch.to(device=self.device)
            y_batch = y_batch.to(device=self.device)
            
            total_samples = 0
            zlist = self.attention_model(X_batch)             # forward pass
            output_tensors = torch.cat(zlist, dim=1).to(device=self.device)
            self.batch_mae = 0
            for i in range(len(X_batch)):
                output = self.regression_model(output_tensors[i]).to(device=self.device)
                mae = torch.mean(torch.abs(output - y_batch[i]))
                self.test_epoch_mae += mae.item() # Accumulate the loss for the entire epoch
                self.batch_mae += mae.item() # Accumulate the loss for the entire batch
                
            avg_mae = self.batch_mae / len(X_batch)  # Calculate the average MAE for the epoch
            
        test_epoch_avg_mae = round((self.test_epoch_mae/len(self.test_dataloader.dataset)),3)
        print("Epoch:", self.epoch, "MAE: ", test_epoch_avg_mae)
        return test_epoch_avg_mae


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build and train a vision transformer for brain age estimation",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Data source path")
    parser.add_argument("dst", help="Save model weights path")
    parser.add_argument("dataShape", type=tuple, help="Shape of a MRI [default 121*145*121]", nargs='?', default=(121, 145, 121), const=(121, 145, 121))
    parser.add_argument("numTrainSamples", type=int, help="Number of training samples [default for OpenBHB is 3172]", nargs='?', default=3172, const=3172)
    parser.add_argument("numTestSamples", type=int, help="Number of test samples [default for OpenBHB is 794]", nargs='?', default=794, const=794)
    parser.add_argument("loadData", type=bool, help="Load the train-test data tensors [default = True]", nargs='?', default=True, const=True)
    parser.add_argument("epochs", type=int, help="No of epochs")
    parser.add_argument("batchSize", type=int, help="Batch size")
    

    args = vars(parser.parse_args())
    src = args["src"]
    dst = args["dst"]
    dataShape = args["dataShape"]
    numTrainSamples = args["numTrainSamples"]
    numTestSamples = args["numTestSamples"]
    loadData = args["loadData"]
    nEpochs = args["epochs"]
    batchSize = args["batchSize"]


    # create/ instantiate the model, optimizer, loss function and trainer
    attention_model = GlobalLocalBrainAge(5, patch_size=64, step=32,
                        nblock=6, backbone='vgg8')
    regression_model = nn.Linear(13, 1).to(device=device)
    attention_model = attention_model.double().to(device=device)
    regression_model = regression_model.double()
    criterion = nn.L1Loss().to(device=device)
    optimizer = optim.Adam(regression_model.parameters(), lr=0.0002)

    # Define the path to save the model after each epoch
    model_dir = dst

    # instantiate the trainer class
    trainer = Trainer(attention_model, optimizer, criterion, device=device)

    if (loadData):
        X_train, X_test,  Y_train, Y_test = trainer.loadFullDataGLT(src, 55,60,src)
    else:
        X_train, X_test,  Y_train, Y_test = ["" for i in range(4)]

    # OpenBHB X_train = (3172,121,145,121,1), X_test = (794,121,145,121,1)
    trainer.trainGLT(model_dir, attention_model, regression_model,criterion,
                      optimizer, nEpochs=nEpochs, 
                     batchSize=batchSize, X_train=X_train, X_test=X_test,  
                     Y_train=Y_train, Y_test=Y_test, loadData=loadData,src=src,
                     numTrainSamples=numTrainSamples, 
                     numTestSamples=numTestSamples,
                     num_workers=1)


if __name__ == '__main__':
    main()