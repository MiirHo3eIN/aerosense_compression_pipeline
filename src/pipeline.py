import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 


# Dataset imports 
import dataset 
# Autoencoder imports 
import ae_model 


train_exp = [3,4,7,8,12,13,16,17,22,23,26,27,31,32,35,36,41,42,45,46,50,51,54,55,60,61,64,65,69,70,73,74,79,80,83,84,88,89,92,93,98,99,102,103,107,108,111,112]
test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113]

path = f"../data/AoA_0deg_Cp/"

def main(): 

    # Import the Cp data with the labels
    
    train_x, train_y  = dataset.TimeseriesSampledCpWithLabels(path, train_exp, 20, 800)
    print(train_x.shape)
    #test_x, test_y  = dataset.TimeseriesSampledCpWithLabels()
    # Compress the data using the choosen compression method
    #   model = ae_model.Autoencoder(model_id)
    #   train_reconstructed, test_reconstructed = data_compression(model, train_x, test_x)


    # Extract the sensitive features from the compressed data deploying the MiniRocket algorithm
    #  train_features, test_features = rocket_feature_extraction(train_rocket = train_reconstructed, test_rocket = test_reconstructed)
    #  del train_reconstructed, test_reconstructed

    # Classification Results on the compressed data
    #  train_features = (train_features, train_y) 
    #  test_features = (test_features, test_y) 
    #  ridge_c_matrix, ridge_acc, ridge_pre = ridge_classifier(train_features, test_features)
    #  rfc_c_matrix, rfc_acc, rfc_pre = random_forest_classifier(train_features, test_features)

    # Save the results in a csv file 


if __name__ == "__main__": 
    main()