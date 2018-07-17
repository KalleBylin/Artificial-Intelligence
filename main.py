import helper
import train
import torch
import os
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def predict():
    return print('To be done')

def test_network():
    return print('To be done')




















flag = False
learn_rate = None
hidden_layers = []
model = None
criterion = None
optimizer = None
actions = ['predict', 'build', 'train', 'test', 'save', 'load', 'exit']

device = helper.get_device()

while (flag != True):
    action = input("\nWhat would you like to do? Actions include: \n{}\n".format(actions))
    if(action == actions[0]):
        if(model != None):
            predict(model)
        else:
            print('\nNo model has been selected \n')

    elif(action == actions[1]):
        model = helper.build_new_network()

    elif(action == actions[2]):
        if(model != None):
            train.train_network(model)
        else:
            print('\nNo model has been selected \n')

    elif(action == actions[3]):
        if(model != None):
            test_network(model)
        else:
            print('\nNo model has been selected \n')

    elif(action == actions[4]):
        if(model != None):
            helper.save_checkpoint(model)
        else:
            print('\nNo model has been selected \n') 

    elif(action == actions[5]):
        model = helper.load_checkpoint()
 
    elif(action == actions[6]):
        flag = False

    else:
        print('Invalid selection. Please select from: \n{}\n'.format(actions))
    












