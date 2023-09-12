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
import logging, os, time, gc
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
from argparse import ArgumentParser

class Trainer:
   def __init__(self, src, dataShape , nEpochs, batchSize, lr, resumeTrain, validate):
      self.src = src
      self.dataShape = dataShape
      self.nEpochs = nEpochs 
      self.batchSize = batchSize
      self.lr= lr
      self.resumeTrain = resumeTrain
      self.validate = validate
      
   def loadData(self):
        # dataShape = (121, 145, 121)

      print("Loading the training data..")

      self.X_train = np.load(self.src + "X_train_VBM_OpenBHB.npy")
      self.Y_train = np.load(self.src + "Y_train_VBM_OpenBHB.npy")

      print("Data loaded having shapes: X-train", self.X_train.shape, " Y-train ", self.Y_train.shape)
      self.steps_per_epoch= self.X_train.shape[0]//self.batchSize

      if self.validate:
         # training-validation split
         self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, random_state=42, test_size=0.2)

         print("Train-val split completed...")
         self.validation_steps = self.X_val.shape[0]//self.batchSize

   def createModel(self):
        
      ############ model hyperparameters #################
      default_parameters = [1e-6,'RawImg','IncludeGender','IncludeScanner',0.00005,0.2,40,10]
      self.decayRate, self.meanImg, self.gender, self.scanner,self.regAmount, self.dropRate, maxAngle,maxShift = default_parameters

      ######### build model #############
      self.model = generateAgePredictionResNet(self.dataShape,regAmount=self.regAmount,dropRate=self.dropRate)

      ######### Initialize checkpoints and early stopping ########
      self.mc = ModelCheckpoint(self.src+'Models\BrainAgeResNet-OpenBHB_VBM.h5',verbose=1,mode='min',save_best_only=True)
      early = EarlyStopping(patience=100, verbose=1)

      # Define the path for saving weights
      self.checkpoint_path_1 = self.src + 'Models\BrainAgeResNet-OpenBHB_VBM-Checkpoint{epoch:02d}.h5'
      if (self.resumeTrain > 0):
         self.model.load_weights(self.checkpoint_path_1.format(epoch=self.resumeTrain))
         print("Loaded previous weights...")


      # Initialize optimizer and loss function
      self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, decay=self.decayRate)
      self.loss_fn = tf.keras.losses.mean_absolute_error

      # Initialize metrics
      self.train_loss_metric = tf.keras.metrics.Mean()
      self.train_mae_metric = tf.keras.metrics.MeanAbsoluteError()
      self.train_mse_metric = tf.keras.metrics.MeanSquaredError()

      # Validation metrics
      self.val_loss_metric = tf.keras.metrics.Mean()
      self.val_mae_metric = tf.keras.metrics.MeanAbsoluteError()
      self.val_mse_metric = tf.keras.metrics.MeanSquaredError()

      # Instantiate the logger callback with the desired log file path
      self.log_file_path = self.src + 'History\ResNet-VBM-Training-log-Epochs{}.txt'.format(self.nEpochs)
      self.history_logger = TrainingHistoryLogger(self.log_file_path)
    
   @tf.function
   def train_step(self, x_batch, y_batch):
      with tf.GradientTape() as tape:
         # Forward pass
         predictions = self.model(x_batch, training=True)
         # Compute the loss
         loss_value = self.loss_fn(y_batch, predictions)
      
      # Compute gradients
      grads = tape.gradient(loss_value, self.model.trainable_variables)
      
      # Update the model's weights
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
      
      # Update metrics
      self.train_loss_metric(loss_value)
      self.train_mae_metric.update_state(y_batch, predictions)
      self.train_mse_metric.update_state(y_batch, predictions)


   @tf.function
   def test_step(self, x_val_batch, y_val_batch):
      val_predictions = self.model(x_val_batch, training=False)
      val_loss_value = self.loss_fn(y_val_batch, val_predictions)
      
      # Update validation metrics
      self.val_loss_metric(val_loss_value)
      self.val_mae_metric.update_state(y_val_batch, val_predictions)
      self.val_mse_metric.update_state(y_val_batch, val_predictions)
      

   def trainingLoop(self):
      print("Fitting the Model...")

      begin = time.time()

      for epoch in range(self.nEpochs):
         print(f"Epoch {epoch + 1}/{self.nEpochs}")
         begin_train = time.time()

         # # Shuffle the training data for each epoch (if needed)
         # permutation = np.random.permutation(self.X_train.shape[0])
         # X_train = self.X_train[permutation]
         # Y_train = self.Y_train[permutation]

         # training loop
         for step in range(self.steps_per_epoch):
            start = step * self.batchSize
            end = (step + 1) * self.batchSize
            
            # Select a batch of data
            x_batch = self.X_train[start:end]
            y_batch = self.Y_train[start:end]
            
            self.train_step(x_batch, y_batch)

         end_train = time.time()    

         # Print training metrics for this epoch
         print(f"Epoch {epoch + 1}/{self.nEpochs}, Loss: {self.train_loss_metric.result()}, MAE: {self.train_mae_metric.result()}, MSE: {self.train_mse_metric.result()}",
               "Time: ", round((end_train-begin_train)/60, 4), "mins")
         
         # Reset metrics
         self.train_loss_metric.result()
         self.train_mae_metric.result()
         self.train_mse_metric.result()

         self.metrics_dict = {
            'Loss': self.train_loss_metric.result(), 
            'MAE': self.train_mae_metric.result(),
            'MSE': self.train_mse_metric.result()
         }


         if self.validate:    
            begin_val = time.time()
            # Validation loop
            for val_step in range(self.validation_steps):
               start = val_step * self.batchSize
               end = (val_step + 1) * self.batchSize
               
               x_val_batch = self.X_val[start:end]
               y_val_batch = self.Y_val[start:end]
               self.test_step(x_val_batch, y_val_batch)
                  
            end_val = time.time()

            # Print validation metrics for this epoch
            print(f"Validation Loss: {self.val_loss_metric.result()}, Validation MAE: {self.val_mae_metric.result()}, Validation MSE: {self.val_mse_metric.result()}",
                  "Time: ", round((end_val-begin_val)/60, 4), "mins")
            
            # Reset metrics
            self.val_loss_metric.reset_states()
            self.val_mae_metric.reset_states()
            self.val_mse_metric.reset_states()

         # Save model weights after each epoch
         self.model.save_weights(self.checkpoint_path_1.format(epoch=epoch + 1))

         self.history_logger.on_epoch_end(epoch, logs=self.metrics_dict)
         self.metrics_dict = {}

      end = time.time()
      print("Training completed after ", round((end-begin)/60,4), " minutes...")

      ######### save the model details (weights, architecture, compilaion details) ############
      self.model.save(self.src + 'Models\BrainAgeResNet-OpenBHB_VBM-TrainedFor{}Epochs.h5'.format(self.nEpochs))
      print("Complete model saved...")

      # free the space of the training and validation data
      del self.X_train, self.Y_train
      if self.validate:
         del self.X_val, self.Y_val
      gc.collect()

   def testModel(self):
      self.model = tf.keras.models.load_model(self.src + 'Models\BrainAgeResNet-OpenBHB_VBM-TrainedFor{}Epochs.h5'.format(self.nEpochs))

      print("Started loading test data...")
      self.X_test = np.load(self.src + "X_test_VBM_OpenBHB.npy")
      self.Y_test = np.load(self.src + "Y_test_VBM_OpenBHB.npy")
      print("Test data loaded having shapes: X-test", self.X_test.shape, " Y-test ", self.Y_test.shape)
      test_steps = self.X_test.shape[0]//self.batchSize

      test_prediction = self.model.predict(self.X_test,
                              verbose=1,
                              max_queue_size=4,
                              batch_size=self.batchSize,
                              workers=1,
                              use_multiprocessing=False
                              )
      print("Test instances estimated...")
      np.save(self.src + "Results\Y_pred-BrainAgeResNet.npy", test_prediction[:,0])
    
      # Evaluate the model on the test set
      test_loss = 0.0
      test_mae = 0.0
      test_mse = 0.0
      for test_step in range(test_steps):
         start = test_step * self.batchSize
         end = (test_step + 1) * self.batchSize
         
         x_test_batch = self.X_test[start:end]
         y_test_batch = self.Y_test[start:end]
         
         test_predictions = self.model(x_test_batch, training=False)
         test_loss_value = self.loss_fn(y_test_batch, test_predictions)
         
         # Accumulate test loss
         test_loss += test_loss_value
         test_mae += tf.reduce_mean(tf.abs(y_test_batch - test_predictions))
         test_mse += tf.reduce_mean(tf.square(y_test_batch - test_predictions))

      # Calculate the average test loss and metrics
      avg_test_loss = test_loss / test_steps
      avg_test_mae = test_mae / test_steps
      avg_test_mse = test_mse / test_steps

      # Print test metrics
      print(f"Test Loss: {avg_test_loss}, Test MAE: {avg_test_mae}, Test MSE: {avg_test_mse}")


class TrainingHistoryLogger(tf.keras.callbacks.Callback):
   def __init__(self, log_file_path):
      super().__init__()
      self.log_file_path = log_file_path
      self.history = {}  # Initialize an empty dictionary to store training metrics

   def on_epoch_end(self, epoch, logs=None):
      # On each epoch end, record the training metrics in the history dictionary
      for key, value in logs.items():
         if key not in self.history:
            self.history[key] = []
         self.history[key].append(value)

      # Save the history to a log file
      with open(self.log_file_path, 'a') as log_file:
         log_file.write(f"Epoch {epoch + 1} Metrics: {logs}\n")


def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser(description="Build and train a ResNet model for brain age prediction",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument("src", help="Source location")
   parser.add_argument("dataShape", type=tuple, help="Shape of a MRI [default 121*145*121]", nargs='?', default=(121, 145, 121), const=(121, 145, 121))
   parser.add_argument("epochs", type=int, help="No of epochs")
   parser.add_argument("batches", type=int, help="No of batches")
   parser.add_argument("lr", type=float, help="Learning rate")
   parser.add_argument("--rt", nargs='?', type=int, default=0, const=0, help="Resume training from previous epochs [default False]")
   parser.add_argument("--validate", action='store_true', help="Perform validation on the validation set [default False]")

   args = vars(parser.parse_args())
   src = args["src"]
   dataShape = args["dataShape"]
   nEpochs = args["epochs"]
   batchSize = args["batches"]
   lr = args["lr"]
   resumeTrain = args["rt"]
   validate = args["validate"]

   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   trainer = Trainer(src, dataShape=dataShape, nEpochs=nEpochs, batchSize=batchSize, lr=lr,
                     resumeTrain=resumeTrain, validate=validate)
   trainer.createModel()
   trainer.loadData()
   trainer.trainingLoop()
   trainer.testModel()


if __name__ == '__main__':
   main()

