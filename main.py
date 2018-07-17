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


def build_network():
    
    # Gets a valid model before proceeding
    partial_model = helper.get_model(None)
        
    # Gets correct learn rate and hidden layers before proceeding
    learn_rate, hidden_layers = helper.get_network_inputs()

    # Buils the network
    model, criterion, optimizer = helper.build_network(learn_rate, hidden_layers, partial_model)
    model.criterion = criterion
    model.optimizer = optimizer

    return model


















flag = False
learn_rate = None
hidden_layers = []
model = None
criterion = None
optimizer = None
actions = ['predict', 'build', 'train', 'test', 'exit']

device = helper.get_device()

while (flag != True):
    action = input("\nWhat would you like to do? Actions include: \n{}\n".format(actions))
    if(action == actions[0]):
        if(model != None):
            predict(model)
        else:
            print('\nNo model has been selected \n')

    elif(action == actions[1]):
        model = build_network()
        
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
        flag = False

    else:
        print('Invalid selection. Please select from: \n{}\n'.format(actions))
    












