# from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
from typing import TypeVar
from dataclasses import dataclass
from enum import Enum

Tensor = TypeVar('torch.tensor')

@dataclass
class ModelParams:
    input_dim:int
    depth:int
    hidden_dim:int
    latent_dim:int
    init_std:float
    dropout_rate:float=0.3
    
@dataclass
class TrainParams:
    n_epochs:int
    batch_size:int
    lkl:float               # lambda Kullback-Leibler divergence
    lc:float                # lambda classification
    learning_rate:float=1.e-3
    patience:int=5
    do_weighted_bce:bool=True
    perc_less_rep_oversample:float=0.0
    bce_pos_weight_mult:float=0.0
