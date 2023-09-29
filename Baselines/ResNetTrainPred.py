#!/usr/bin/env python

import torch 
import torch.nn as nn
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from resnet import resnet18
import torch.optim as optim
import argparse
from argparse import ArgumentParser
import time

import warnings
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, batchSize, lr, nEpochs, device, validate):

        self.device = device
        self.batchSize = batchSize
        self.lr = lr
        self.nEpochs = nEpochs
        self.validate = validate
        


    def buildModel(self, Model, criterion, optimizer, model_dir):
        self.model_dir = model_dir
        self.criterion = criterion
        self.optimizer = optimizer
        self.resnetModel = Model
        self.prevEpochs = 0


    def createDataLoader(self, source_path):
        """
        - loads the dataset and creates data-loaders for the training and test set
        """
        print("Started loading data..")
        self.X_train = torch.from_numpy(np.load(source_path + "X_train_VBM_OpenBHB.npy")).permute(0,4,1,2,3)
        self.Y_train = np.load(source_path + "Y_train_VBM_OpenBHB.npy")

        self.X_test = torch.from_numpy(np.load(source_path + "X_test_VBM_OpenBHB.npy")).permute(0,4,1,2,3)
        self.Y_test = np.load(source_path + "Y_test_VBM_OpenBHB.npy")

        print("Data tensor loaded having shapes:> X-train: ", self.X_train.shape, 
            "Y-train: ", self.Y_train.shape)
   
        self.train_dataloader = DataLoader(list(zip(self.X_train, self.Y_train)), 
                                           batch_size=self.batchSize,  shuffle=True)
        self.test_dataloader = DataLoader(list(zip(self.X_test, self.Y_test)), 
                                          batch_size=self.batchSize,  shuffle=True)
        print("Created train-test data loaders...")

    def loadCheckpoint(self, model, optimizer, load_path, model_dir):
        """
        - To resume training, restore the model and optimizer's state dict
        - Returns the model and optimizer's state dict and number of previously 
        trained epochs (total number of epochs + previously trained epochs)
        """
        # model.load_state_dict(torch.load(load_path))
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prevEpochs = checkpoint['epoch']
        
        print("Model checkpoint loaded from Epoch: ", prevEpochs)
        self.resnetModel = model
        self.optimizer = optimizer
        self.nEpochs = self.nEpochs + prevEpochs
        self.prevEpochs = prevEpochs+1
        self.model_dir = model_dir
    
    def trainModel(self):
        self.train_loss = [] 
        self.val_acc = []

        # Loop through the number of epochs
        for self.epoch in range(self.prevEpochs, self.nEpochs):
            begin = time.time()

            self.train_epoch_loss = 0  # Initialize loss for the entire epoch
            self.val_epoch_mae = 0   # Initialize the total MAE for the entire epoch
            self.optimizer.zero_grad()  # a clean up step for PyTorch

            loss = self.trainEpoch()
            self.train_loss.append(loss)

            if (self.validate):
                mae = self.evaluateEpoch()
                self.val_acc.append(mae)

            time.sleep(1)
            # store end time
            end = time.time()
            print("Training epoch ", self.epoch, " completed in ", 
                  round((end - begin)/60, 2), " minutes..")
            
        return self.train_loss , self.val_acc
    

    def trainEpoch(self):
        """
        Training loop -> train the model using the training set for one complete epoch 
        """   
        print("Training epoch ",self.epoch," started.." )

        # Loop through the batches and update the gradients after each batch
        for X_batch, y_batch in self.train_dataloader:
            # Obtaining the cuda parameters
            X_batch = X_batch.to(device=self.device)
            y_batch = y_batch.to(device=self.device)
            
            self.resnetModel.train()
            self.batch_loss = 0
            # loop through each sample in the current batch 
            for i in range(len(X_batch)):
                sample = X_batch[i]
                sample = sample.reshape((1,)+sample.shape)

                # Forward propagation
                output = self.resnetModel(sample)
                output = output[0].reshape([1, -1]).to(device=self.device)
                # output is a tensor containing the soft labels (83 values) of the particular subject

                # Compute the loss for each sample
                loss = criterion(output, y_batch[i])

                self.batch_loss += loss  # Accumulate the loss for the entire batch
                self.train_epoch_loss += loss  # Accumulate the loss for the entire epoch
                
            batch_avg_loss = self.batch_loss / len(X_batch)  # Calculate the average loss for the batch
        
            self.optimizer.zero_grad()  # Clear accumulated gradients

            batch_avg_loss.backward()  # Backward pass (compute parameter updates)
            # print("Batch completed")
            self.optimizer.step()


        train_epoch_avg_loss = round((self.train_epoch_loss / len(self.train_dataloader.dataset)).item(),3)
        print("Epoch:", self.epoch, "Loss:", train_epoch_avg_loss)
        # Save the model's state dictionary after each epoch
        model_path = f'{self.model_dir}ResNet18-Model-epoch-{self.epoch}.pth'
        # torch.save(self.resnetModel.state_dict(), model_path)
        self.saveCheckpoint(self.resnetModel,self.optimizer,model_path,
                            self.epoch,train_epoch_avg_loss )

        return train_epoch_avg_loss

    def saveCheckpoint(self, model, optimizer, save_path, epoch, loss):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }, save_path)
        print("Model saved...")



    def checkAccuracy(self):
        """
        Validation loop: evaluate the model on the entire test set for one epoch
        """
        self.test_epoch_mae = 0  
        print("Testing the model .." )
        self.resnetModel.eval()  # Set the model to evaluation mode
        with torch.no_grad(): # we don't compute gradients here 
            # loop through each batch of the test set and compute the evaluation metrics
            for X_batch, y_batch in self.test_dataloader:
               
                # Obtaining the cuda parameters
                X_batch = X_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                
                total_samples = 0
                self.batch_mae = 0

                for i in range(len(X_batch)):
                    sample = X_batch[i]
                    sample = sample.reshape((1,)+sample.shape)
                
                    output = self.resnetModel(sample)
                    output = output[0].reshape([1, -1])

                    x = output.detach().cpu().numpy().reshape(-1)
                    y = y_batch[i].detach().cpu().numpy().reshape(-1)

                    mae = torch.mean(torch.abs(output - y_batch[i]))
                    self.test_epoch_mae += mae.item() # Accumulate the loss for the entire epoch
                    self.batch_mae += mae.item() # Accumulate the loss for the entire batch
                    
                avg_mae = self.batch_mae / len(X_batch)  # Calculate the average MAE for the epoch
                
            test_epoch_avg_mae = round((self.test_epoch_mae/len(self.test_dataloader.dataset)),3)
            print("MAE on the test set: ", test_epoch_avg_mae, " years")
            



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build and train a 3D CNN for brain age estimation",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="To load train-test data from source location")
    parser.add_argument("dst", help="To save model files to the dest dir")
    parser.add_argument("dataShape", type=tuple, help="Shape of a MRI [default 121*145*121]", nargs='?', default=(121, 145, 121), const=(121, 145, 121))
    parser.add_argument("epochs", type=int, help="No of epochs")
    parser.add_argument("batches", type=int, help="No of batches")
    parser.add_argument("lr", type=float, help="Learning rate")
    parser.add_argument("--rt",  nargs='?', default="", const="", type=str, help="Path of the previously trained (.pth) model file")
    parser.add_argument("--validate", action='store_true', help="Perform validation on the validation set [default False]")
    parser.add_argument("--test", action='store_true', help="Perform validation on the validation set [default False]")

    args = vars(parser.parse_args())
    source_path = args["src"]
    model_dir = args["dst"]
    dataShape = args["dataShape"]
    nEpochs = args["epochs"]
    batchSize = args["batches"]
    lr = args["lr"]
    trainedModelFile = args["rt"]
    validate = args["validate"]
    test = args["test"]

    # create/ instantiate the model, optimizer, loss function and trainer

    resnetModel = resnet18().to(device=device)
    resnetModel = resnetModel.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnetModel.parameters(), lr=lr) 

    # instantiate the trainer class
    trainer = Trainer(batchSize=batchSize, lr=lr, nEpochs=nEpochs, 
                      device=device, validate=validate)
    if (len(trainedModelFile)==0):
        trainer.buildModel(Model=resnetModel, criterion=criterion,
                        optimizer=optimizer, model_dir=model_dir)
    else:
        trainer.loadCheckpoint(model=resnetModel, optimizer=optimizer,
                                load_path=trainedModelFile, model_dir=model_dir)
    trainer.createDataLoader(source_path=source_path)
    trainer.trainModel()
    if (test):
        trainer.checkAccuracy()



