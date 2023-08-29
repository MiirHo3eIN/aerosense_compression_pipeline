
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn 


# Dataset imports 
import dataset 
# Autoencoder imports 
import ae_model 
import utils

import configs 
from configs import Model_configs, Dataset_configs



def main(dataset_config, model_config): 
    
    # Import the Cp data with the labels
    
    train , y_train = dataset.TimeseriesSampledCpWithLabels(dataset_config.path, dataset_config.train_exp, 10, 800)
    test , y_test = dataset.TimeseriesSampledCpWithLabels(dataset_config.path, dataset_config.test_exp, 10, 800)

    # Compress the data using the choosen compression method
    _ , test_reconstructed = utils.data_compression(model_config, train , test )

    # Extract the sensitive features from the compressed data deploying the MiniRocket algorithm
    train_features, test_features, test_original_features =utils.rocket_feature_extraction(train, test_reconstructed, test)
    del test_reconstructed
    del train
    del test


    ds_train_mean = train_features.mean()
    ds_train_std = train_features.std()
    ds_test_mean = test_features.mean()
    ds_test_std = test_features.std()
    ds_orig_mean = test_original_features.mean()
    ds_orig_std = test_original_features.std()

    df_train = pd.concat([ds_train_mean,ds_train_std], axis=1)
    df_test= pd.concat([ds_test_mean, ds_test_std], axis=1)
    df_orig= pd.concat([ds_orig_mean, ds_orig_std], axis=1)

    df_diff_recon= df_train.sub(df_test).abs()
    df_diff_original= df_train.sub(df_orig).abs()
    
    fig, axes = plt.subplots(1, 2)
    title = '''Somewhat complicated Plot:
Features were extracted for train, test, and reconstructed test sets. Then for all sets all features were calculated.
Then the mean and std for each feature and ech set wer calculated. Then the absolute difference between the train set and test sets  for each feaure was calculated.
To show the "distribution" the resulting differences are plotted as a histogram.'''
    fig.suptitle(title)

    axes[0].set_title('With unaltered Testdata (original)')
    sns.histplot(ax=axes[0], data=df_diff_original, x=0, binwidth=0.002,)
    axes[0].set_xlim(0,0.2)
    axes[0].set_yscale('log')
    axes[0].set_ylim(0,10000)
    axes[0].set_xlabel("Difference of the means of the features")


    axes[1].set_title('With Reconstructed Testdata')
    sns.histplot(ax=axes[1], data=df_diff_recon, x=0, binwidth=0.002)
    axes[1].set_xlim(0,0.2)
    axes[1].set_yscale('log')
    axes[1].set_ylim(0,10000)
    axes[1].set_xlabel("Difference of the means of the features")

    plt.show()




if __name__ == "__main__": 
    
    row = utils.modelChooser()

    model_config = Model_configs(
        path_models = "../trained_models/",
        model_id = row['model_id'],
        arch_id = row['arch_id'],
        seq_len = row['window_size']
    )

    dataset_config = Dataset_configs(
        path = f"../data/cp_data/AoA_0deg_Cp/",
        seq_len = row['window_size'],
        train_exp = [3,4,7,8,12,13,16,17,22,23,26,27,31,32,35,36,41,42,45,46,50,51,54,55,60,61,64,65,69,70,73,74,79,80,83,84,88,89,92,93,98,99,102,103,107,108,111,112],
        # train_exp = [3,4,7,8,112],
        test_exp = [5,9,14,18,24,28,33,37,43,47,52,56,62,66,71,75,81,85,90,94,100,104,109,113])
        # test_exp = [5,9,14,113])
        
        
    main(dataset_config , model_config)

