# Brain Age Estimation with Ensemble-Based Deep Feature Extraction

![Project Image](path/to/your/project/image.png) <!-- If you have a project image, include it here -->

## Overview

This repository contains the implementation of an ensemble-based deep feature extraction method for training a brain age estimation model. The model predicts the brain age based on raw voxel features or 3D volumes of brain MRIs. The project involves fusing the results of 4 state-of-the-art deep learning-based brain age estimation models. The proposed framework demonstrates improved performance compared to individual brain age estimation models. The implementation is done using both TensorFlow and PyTorch.

## Features

- Brain age estimation using raw voxel features or 3D volumes of brain MRIs.
- Fusion of results from 4 state-of-the-art deep learning-based models.
- Improved performance through ensemble-based approach.
- Implementation using both TensorFlow and PyTorch.

## Dataset
The proposed brain age estimation framework is developed using the healthy MRIs from a novel benchmark dataset for brain age estimation, [Open Big Healthy Brain (OpenBHB)](https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing). We used T1-weighted 3D brain volumes of 3966 healthy samples from the OpenBHB dataset which are gathered from 10 public datasets (IXI, ABIDE 1, ABIDE 2, CoRR, GSP, Localizer, MPI-Leipzig, NAR, NPC, RBP) acquired across 93 different centers, spread worldwide (North America, Europe and China). The volumes are  uniformly pre-processed with CAT12 (SPM) and Quasi-Raw (in-house minimal pre-processing), with ages ranging from 6 to 88 years old, balanced between males and females. We followed a 80:20 train-test split and employed a 10-fold cross validation strategy on the training set.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mansoorbalouch/Deep-Feature-Ensemble-Based-BrainAGE.git
   cd your-repo

## Usage

- Data Preparation: Prepare your MRI data as per the required format. Provide necessary paths in the code.
- Training: Run the training script for individual models using TensorFlow and PyTorch.
- Ensemble: Implement the ensemble method as described in the project.
- Inference: Use the trained ensemble model for brain age estimation.

## Results
- The evaluation metrics of Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R^2) are used to assess the brain age estimation models. 
- The state-of-the-art models were able to achieve an MAE of 2.14 years while our proposed ensemble-based approach achieves an MAE of 2 years, indicating a significant improvement in the brain age estimation accuracy.