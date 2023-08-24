import numpy as np 
import pandas as pd

from dataclasses import dataclass


@dataclass
class Dataset_configs:
    path: str
    seq_len: int
    train_exp: list
    test_exp: list

@dataclass
class Model_configs:
    path_models: str
    arch_id: str
    