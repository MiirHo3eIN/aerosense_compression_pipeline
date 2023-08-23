import numpy as np 
import pandas as pd

from dataclasses import dataclass


@dataclass
class Dataset_configs:
    path: str
    seq_len: int