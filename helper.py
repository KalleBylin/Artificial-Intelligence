import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
import pandas as pd
import seaborn as sb
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import constants as Constant
from collections import OrderedDict
import time
from PIL import Image
from torch.autograd import Variable

# Gets the input from the user to build the neural network
def get_network_inputs():
    
    flag = False
    while (flag == False):
    
        learn_rate = input("\n\nPlease enter the learning rate\n") 
        hidden_layers = input("\nPlease enter the number of hidden layers seperated by commas \nThe final hidden layer will be taken as the output layer \n Eg: 5,6,3,2 \n")

        try:
            learn_rate = int(learn_rate)
            hidden_layers = list(map(int, hidden_layers.split(',')))
            flag = True
            print("\nLearning Rate = {} and Number of Hidden Layers = {}".format(learn_rate, hidden_layers))
            return(learn_rate, hidden_layers)
        except ValueError:
            print('\nPlease enter numbers only')

# Gets a model from a predefined list
def get_model():
    ls_models = ["densenet161", "vgg16"]
    
    flag = False
    while (flag == False):
        model = input("\n\nPlease type the required model \n Models include: \n {} \n".format(ls_models))
        
        if(model == ls_models[0]):
            print("Selected Model = {}".format(ls_models[0]))
            return (models.densenet161(pretrained=True), 1024)
        elif(model == ls_models[1]):
            print("Selected Model = {}".format(ls_models[1]))
            return (models.vgg16(pretrained=True), 25088)
        else:
            print('\nInvalid model \n"Please type the required model \n Models include: \n {} \n'.format(ls_models))

# Builds a neural network
def build_network(learn_rate, input_layer, hidden_layers, partial_model):
    
    print("\nBuilding neural network")
    drop_rate = 0.5

    # prevents backpropogation through the pretrained network, hence we are only training our classifier
    for param in partial_model.parameters():
        param.requires_grad = False

    classifier = nn.ModuleList([nn.Linear(input_layer, hidden_layers[0])])

    print('Range = {}'.format(len(hidden_layers) - 1))
    for i, item in enumerate(hidden_layers):    
        
        if(i != (len(hidden_layers) - 2)):
            classifier.append(nn.Linear(item, hidden_layers[i+1]))
            classifier.append(nn.Dropout(p=drop_rate))
        else:
            classifier.append(nn.Linear(item, hidden_layers[i+1]))
            classifier.append(nn.LogSoftmax(dim=1))
            print('\n----------------- CLASSIFIER -----------------')
            print(classifier)
            partial_model.classifier = classifier
            
            criterion = nn.NLLLoss()

            # only train the classifier parameters, feature parameters are frozen
            optimizer = optim.Adam(partial_model.classifier.parameters(), lr=learn_rate)
            
            print('\nNeural network built')
            return (partial_model, criterion, optimizer)

# Transforms images, creates and returns dataloader
def get_dataloader(type, data_dir):
    transforms = None
    loader = None
    if(type == Constant.TRAIN):
        transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                        ])

        data = datasets.ImageFolder(data_dir, transform=transforms)
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
        return loader
    elif(type == Constant.TEST):
        transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

        data = datasets.ImageFolder(data_dir, transform=transforms)
        loader = torch.utils.data.DataLoader(test_data, batch_size=32)
        return loader
    elif(type == Constant.VALID):
        transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

        data = datasets.ImageFolder(data_dir, transform=transforms)
        loader = torch.utils.data.DataLoader(test_data, batch_size=32)
        return loader
    else:
        print('\n\nInvalid Action Type, Please select TRAIN, TEST or VALID')
        return False

# Loads JSON file with a dictionary mapping the integer encoded classes to actual class names
def map_labels():
    flag = False
    while(flag == False):
        filePath = input('\nPlease enter path to label map JSON file from current location\n')
        if(os.path.exists(filePath)):
            with open(filePath, 'r') as f:
            idx_to_class = json.load(f)  
            flag = True
            return idx_to_class
    
