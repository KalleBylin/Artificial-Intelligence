import helper
import torch
import os
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Trains the neural network
def train_network(model):
    flag = False
    epochs = 0
    
    while (flag == False):
        try:
            epochs = int(input("\n\nPlease enter the epochs for training\n") )
            data_dir = input('\nPlease enter path to data from current location\n')
            if(os.path.isdir(data_dir)):
                flag = True
            else:
                print('\nInvalid path.. No such folder exists')
        except ValueError:
            print('\nPlease enter numbers only')

    
