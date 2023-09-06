#!/usr/bin/env python

"""
dependencies:
- NVIDIA driver v536
- CUDA toolkit v11.2
- cuDNN library v8.1
- tensorflow v2.10
- python v3.10
"""

import warnings
warnings.simplefilter("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
from tensorflow.keras.models import load_model
from collections import defaultdict
from tensorflow.keras.optimizers import Adam, SGD,Adagrad
from tensorflow.compat.v1 import reset_default_graph
import ResNet
from ResNet import generateAgePredictionResNet
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping 
from tensorflow.python.keras import backend as K
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
from argparse import ArgumentParser


# Parse command line arguments
parser = argparse.ArgumentParser(description="Build and train a ResNet model for brain age prediction",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("src", help="Source location")
parser.add_argument("dataShape", type=tuple, help="Shape of a MRI [default 121*145*121]", nargs='?', default=(121, 145, 121), const=(121, 145, 121))
parser.add_argument("epochs", type=int, help="No of epochs")
parser.add_argument("batches", type=int, help="No of batches")

args = vars(parser.parse_args())
src = args["src"]
dataShape = args["dataShape"]
nEpochs = args["epochs"]
batchSize = args["batches"]

# dataShape = (121, 145, 121)

print("Loading the data..")

X_train = np.load(src + "X_train_VBM_OpenBHB.npy")
X_test = np.load(src + "X_test_VBM_OpenBHB.npy")
Y_train = np.load(src + "Y_train_VBM_OpenBHB.npy")
Y_test = np.load(src + "Y_test_VBM_OpenBHB.npy")

print("Data loaded having shapes: X-train", X_train.shape, " X-test ", X_test.shape)

############ model hyperparameters #################
steps_per_epoch= X_train.shape[0]//batchSize
validation_steps = X_test.shape[0]//batchSize
default_parameters = [0.001,1e-6,'RawImg','IncludeGender','IncludeScanner',0.00005,0.2,40,10]
lr, decayRate, meanImg, gender, scanner,regAmount, dropRate, maxAngle,maxShift = default_parameters

######### build model #############
model = generateAgePredictionResNet(dataShape,regAmount=regAmount,dropRate=dropRate)
lr=0.001
#decayRate=1e-6
#momentum=0.9
adam = Adam(learning_rate=lr)
model.compile(loss='mean_absolute_error',optimizer=adam, metrics=['mae','mse'])
print("Model compiled...")

######### save the best model ########
mc = ModelCheckpoint(src+'/Models/BrainAgeResNet-OpenBHB_VBM.h5',verbose=1,mode='min',save_best_only=True)
early = EarlyStopping(patience=100, verbose=1)

print("Fitting the Model...")
######### train the model ##############
h = model.fit(x=X_train, y=Y_train,
                        validation_steps=validation_steps,
                        steps_per_epoch=steps_per_epoch, 
                        epochs=nEpochs,
                        batch_size=batchSize,
                        verbose=1,
                        max_queue_size=32,
                        callbacks=[mc,early]
                           )
print("Model fitted...")

########## save the model details (weights, architecture, compilaion details) ############
model.save(src + '/Models/BrainAgeResNet-OpenBHB_VBM-TrainedFor{}Epochs.h5'.format(len(h.history['loss'])))
print("Model saved...")

######### load the best model and perform predictions on the val and test set #######
model = load_model(src + 'Models/BrainAgeResNet-OpenBHB_VBM-TrainedFor{}Epochs.h5'.format(len(h.history['loss'])))
# val_prediction = model.predict(X_val,
#                         verbose=1,
#                         max_queue_size=32,
#                         workers=4,
#                         use_multiprocessing=False
#                         )

test_prediction = model.predict(X_test,
                        verbose=1,
                        max_queue_size=32,
                        workers=4,
                        use_multiprocessing=False
                        )

model_performance = model.evaluate(X_test, Y_test)
print(model_performance)
# train_predictions[:,0].save(src + "Y_val_pred-BrainAgeResNet")
np.save(src + "Y_test_pred-BrainAgeResNet.npy", test_prediction[:,0])


