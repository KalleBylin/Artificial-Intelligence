import helper
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def predict():
    return print('To be done')

def train_network():
    return print('To be done')

def test_network():
    return print('To be done')


def build_network(device):
    
    # Gets a valid model before proceeding
    partial_model, input_layer = helper.get_model()
        
    # Gets correct learn rate and hidden layers before proceeding
    learn_rate, hidden_layers = helper.get_network_inputs()

    model, criterion, optimizer = helper.build_network(learn_rate, input_layer, hidden_layers, partial_model)
    
    # load the model to the device GPU / CPU
    model.to(device)

    return model, criterion, optimizer


















flag = False
learn_rate = None
hidden_layers = []
model = None
criterion = None
optimizer = None
actions = ['predict', 'build', 'train', 'test']

# checks if GPU is available else runs on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO Change output layer to one that can be entered by a user
output_layer = 102

while (flag != True):
    action = input("\nWhat would you like to do? Actions include: \n{}\n".format(actions))
    flag = True
    if(action == actions[0]):
        if(model != None):
            predict(model)
        else:
            print('\nNo model has been selected \n')

    elif(action == actions[1]):
        model, criterion, optimizer = build_network(device)
        
    elif(action == actions[2]):
        if(model != None):
            train_network(model)
        else:
            print('\nNo model has been selected \n')

    elif(action == actions[3]):
        if(model != None):
            test_network(model)
        else:
            print('\nNo model has been selected \n')

    else:
        print('Invalid selection. Please select from: \n{}\n'.format(actions))
        flag = False












