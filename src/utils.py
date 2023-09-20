import numpy as np 
import pandas as pd

from simple_term_menu import TerminalMenu
# Rocket impports 
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# Classifiers imports 
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier

# Classifications Accuracy imports
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Import Torch 
import torch
import torch.nn as nn

# Custom imports 
import ae_model 

import configs 
from configs import Model_configs

# To save the trained model
from csv import DictWriter


import os

def rocket_feature_extraction(train_rocket, test_rocket, test_original): 

    minirocket_multi = MiniRocketMultivariate()
    minirocket_multi.fit(train_rocket.numpy())
    X_train_transform = minirocket_multi.transform(train_rocket.numpy())
    X_test_transform = minirocket_multi.transform(test_rocket.numpy())
    X_test_transform_original = minirocket_multi.transform(test_original.numpy())

    return X_train_transform, X_test_transform, X_test_transform_original


def ridge_classifier(train_features, y_train, test_features, y_test): 

    X_train_transform = train_features
    X_test_transform  = test_features


    ridge_clf = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17) ) 
    ridge_clf.fit(X_train_transform, y_train)
    y_pred = ridge_clf.predict(X_test_transform)

    ridge_acc = accuracy_score(y_test, y_pred)
    ridge_pre = precision_score(y_test, y_pred, average='weighted')
    ridge_c_matrix = confusion_matrix(y_test, y_pred)

    return ridge_c_matrix, ridge_acc, ridge_pre

def random_forest_classifier(train_features, y_train, test_features, y_test): 

    X_train_transform = train_features
    X_test_transform  = test_features

    rfc = RandomForestClassifier(n_estimators=800, max_features="sqrt")
    rfc.fit(X_train_transform, y_train)
    y_pred = rfc.predict(X_test_transform)

    rfc_acc = accuracy_score(y_test, y_pred)
    rfc_pre = precision_score(y_test, y_pred, average='weighted')
    rfc_c_matrix = confusion_matrix(y_test, y_pred)

    return rfc_c_matrix, rfc_acc, rfc_pre


# These two should be merged into a Class 
def torch_eval(model,x_input: torch.Tensor) -> torch.Tensor:

    with torch.no_grad():
        model.eval() 
        x_hat = model(x_input.float())
    return x_hat

def data_compression(model_configs, train_x, test_x): 
    # Call the model to compress the data 

    model = ae_model.Model(model_configs.arch_id)
    
    model_id = model_configs.model_id
    model.load_state_dict(torch.load(f"{model_configs.path_models}{model_id}.pt"))  

    reconstructed_train = torch_eval(model, train_x)
    reconstructed_test = torch_eval(model, test_x)

    return reconstructed_train , reconstructed_test 


def _is_file_exist(file_name: str) -> bool:
    return os.path.isfile(file_name)

def write_to_csv(data: dict, file_name: str) -> None: 
    write_dir = f"../{file_name}.csv"
    if (_is_file_exist(write_dir)): 
        _append_to_csv(data, file_name)
    else: 
        _create_csv(data, file_name)

def _append_to_csv(data: dict, file_name: str) -> None:
    
    print("Appending to the pipeline results csv file")
    print("++"*15)
    with open(f"../{file_name}.csv", "a") as FS:
        
        headers = list(data.keys())

        csv_dict_writer = DictWriter(FS, fieldnames = headers) 

        csv_dict_writer.writerow(data)

        FS.close()


def loadFileToDf(file):

    # if file is none the default file is used
    if file is None: file = "../training_results.csv"

    # check if file exists
    if not os.path.isfile(file): raise Exception()

    return pd.read_csv(file)

def modelChooser(file=None):
    df = loadFileToDf(file)
    options = []
    model_ids = []
    for row in df.iterrows():
        x = row[1]
        model_ids.append(x['model_id'])
        options.append(f"{x['model_id']} - {x['window_size']} -> {x['latent_channels']} x {x['latent_seq_len']}")
    terminal_menu = TerminalMenu(options)
    menu_entry_index = terminal_menu.show()

    model_id = model_ids[menu_entry_index]
    return df.loc[df['model_id'] == model_id].to_dict(orient='index')[menu_entry_index]

def _create_csv(data: dict, file_name: str) -> None:
    df = pd.DataFrame.from_dict(data, orient = "index").T.to_csv(f"../{file_name}.csv", header = True, index = False)
    print("Created the csv file is as follows:")
    print(df)
