import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 


# Dataset imports 
import dataset 
# Autoencoder imports 
import ae_model 
import utils

import configs 
from configs import Model_configs, Dataset_configs


dataset_config = Dataset_configs(
    path = f"../data/AoA_0deg_Cp/",
    seq_len = 800,
    train_exp = [3,4,7,8,12,13,16,17,22,23,26,27,31,32,35,36,41,42,45,46,50,51,54,55,60,61,64,65,69,70,73,74,79,80,83,84,88,89,92,93,98,99,102,103,107,108,111,112],
    test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]
)




def main(): 
    
    # Import the Cp data with the labels
    
    train  = dataset.TimeseriesSampledCpWithLabels(dataset_config.path, dataset_config.train_exp, 20, 800)
    test  = dataset.TimeseriesSampledCpWithLabels(dataset_config.path, dataset_config.test_exp, 20, 800)

    # Compress the data using the choosen compression method
    model_id = 'a61c'
    model = ae_model.Model(model_id)
    train_reconstructed, test_reconstructed = utils.data_compression(model, train[0], test[0])
    train_y = train[1]
    test_y = test[1]
    del train, test


    # Extract the sensitive features from the compressed data deploying the MiniRocket algorithm
    train_features, test_features = utils.rocket_feature_extraction(train_rocket = train_reconstructed, test_rocket = test_reconstructed)
    del train_reconstructed, test_reconstructed

    # Classification Results on the compressed data
    train_features = (train_features, train_y) 
    test_features = (test_features, test_y) 
    ridge_c_matrix, ridge_acc, ridge_pre = utils.ridge_classifier(train_features, test_features)
    rfc_c_matrix, rfc_acc, rfc_pre = utils.random_forest_classifier(train_features, test_features)

    # Save the results in a csv file 
    # TBD

if __name__ == "__main__": 
    main()