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


clasification_accuracy = {
    "model_id": [],
    "arch_id": [],  
    "ridge": [],
    "rfc": [],
}

precision_results = {
    
    "model_id": [],
    "arch_id": [],  
    "ridge": [],
    "rfc": [],
}




def data_logger(model_config, ridge_acc, rfc_acc, ridge_pre, rfc_pre): 

    clasification_accuracy["model_id"].append(model_config.model_id)
    clasification_accuracy["arch_id"].append(model_config.arch_id)
    clasification_accuracy["ridge"].append(ridge_acc)
    clasification_accuracy["rfc"].append(rfc_acc)

    precision_results["model_id"].append(model_config.model_id)
    precision_results["arch_id"].append(model_config.arch_id)
    precision_results["ridge"].append(ridge_pre)
    precision_results["rfc"].append(rfc_pre)

    utils.write_to_csv(clasification_accuracy, "classification_accuracy")
    utils.write_to_csv(precision_results, "precision_results")




def main(dataset_config, model_config): 
    
    # Import the Cp data with the labels
    
    train  = dataset.TimeseriesSampledCpWithLabels(dataset_config.path, dataset_config.train_exp, 20, dataset_config.seq_len)
    test  = dataset.TimeseriesSampledCpWithLabels(dataset_config.path, dataset_config.test_exp, 20, dataset_config.seq_len)

    # Compress the data using the choosen compression method

    train_reconstructed, test_reconstructed = utils.data_compression(model_config, train[0], test[0])
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
    data_logger(model_config, ridge_acc, rfc_acc, ridge_pre, rfc_pre)

if __name__ == "__main__": 
    
    # Let's read all the avaiable trained models and run the pipeline for each of them  
    models = pd.read_csv("../training_results.csv", usecols=["model_id", 'arch_id', "window_size"])

    for _, row in models.iterrows():
        print("Processing")
        print(row['model_id'], row['arch_id'], row['window_size'])
        
        model_config = Model_configs(
            path_models = "../trained_models/",
            model_id = row['model_id'],
            arch_id = row['arch_id'],
            seq_len = row['window_size']
        )

        dataset_config = Dataset_configs(
        path = f"../data/AoA_0deg_Cp/",
        seq_len = row['window_size'],
        train_exp = [3,4,7,8,12,13,16,17,22,23,26,27,31,32,35,36,41,42,45,46,50,51,54,55,60,61,64,65,69,70,73,74,79,80,83,84,88,89,92,93,98,99,102,103,107,108,111,112],
        test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113])
        
        
        main(dataset_config , model_config)