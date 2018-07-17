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
import os 
import helper
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
            learn_rate = float(learn_rate)
            hidden_layers = list(map(int, hidden_layers.split(',')))
            flag = True
            print("\nLearning Rate = {} and Number of Hidden Layers = {}".format(learn_rate, hidden_layers))
            return(learn_rate, hidden_layers)
        except ValueError:
            print('\nPlease enter numbers only')

# Gets a model from a predefined list
def get_model(model):
    ls_models = ["densenet161", "vgg16"]
    flag = False
    while (flag == False):
        if(model == None):
            model = input("\n\nPlease type the required model \n Models include: \n {} \n".format(ls_models))

        if(model == ls_models[0]):
            print("\nSelected Model = {}\n".format(ls_models[0]))
            model = models.densenet161(pretrained=True)
            model.name = 'densenet161'
            model.input_layer = 1024
            return (model)
        elif(model == ls_models[1]):
            print("\nSelected Model = {}\n".format(ls_models[1]))
            model = models.vgg16(pretrained=True)
            model.name = 'vgg16'
            model.input_layer = 25088
            return (model)
        else:
            model = None
            print('\nInvalid model \n"Please type the required model \n Models include: \n {} \n'.format(ls_models))

# Builds a neural network
def build_network(learn_rate, hidden_layers, partial_model):
    
    print("\nBuilding neural network")
    drop_rate = 0.5
    partial_model.learn_rate = learn_rate
    partial_model.drop_rate = drop_rate
    partial_model.hidden_layers = hidden_layers

    # prevents backpropogation through the pretrained network, hence we are only training our classifier
    for param in partial_model.parameters():
        param.requires_grad = False

    classifier = nn.ModuleList([nn.Linear(partial_model.input_layer, hidden_layers[0])])

    print('\nLinking the neurons. {} layers found'.format(len(hidden_layers)))
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
            
            partial_model = helper.load_device(partial_model)
            print('\nNeural network built')
            return (partial_model, criterion, optimizer)

# Transforms images, creates and returns dataloader
def get_dataloader(data_dir, type):
    loader = None
    if(type == Constant.TRAIN):
        transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

        data = datasets.ImageFolder(data_dir, transform=transform)
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
        return loader
    elif(type == Constant.TEST):
        transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

        data = datasets.ImageFolder(data_dir, transform=transform)
        loader = torch.utils.data.DataLoader(data, batch_size=32)
        return loader
    elif(type == Constant.VALID):
        transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

        data = datasets.ImageFolder(data_dir, transform=transform)
        loader = torch.utils.data.DataLoader(data, batch_size=32)
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

# Save the checkpoint 
def save_checkpoint(model, class_to_idx):
    
    checkpoint = {'hidden_layers': model.hidden_layers,
                    'state_dict': model.state_dict(),
                    'idx_to_class': idx_to_class,
                    'drop_rate':model.drop_rate,
                    'name':model.name,
                    'input_layer':model.input_layer,
                    'learn_rate':model.learn_rate
                 }
    print("\nSaving checkpoint\n")
    return torch.save(checkpoint, 'checkpoint.pth')    

# Loads a pre-trained model from a checkpoint
def load_checkpoint():

    filepath = get_file_path("\nPlease enter path of model checkpoint\n")
    print("\nLoading model\n")
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    partial_model = get_model(checkpoint['name'])
    partial_model.hidden_layers = checkpoint['hidden_layers']
    partial_model.drop_rate = checkpoint['drop_rate']
    partial_model.name = checkpoint['name']
    partial_model.input_layer = checkpoint['input_layer']
    partial_model.learn_rate = checkpoint['learn_rate']

    model, criterion, optimizer = build_network(partial_model.learn_rate, partial_model.hidden_layers, partial_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.idx_to_class = checkpoint['idx_to_class']
    model.criterion = criterion
    model.optimizer = optimizer
    
    return (model)
    

# Does a validation pass on the model using the validation loader
def validation(model, validloader, criterion):
    device = helper.get_device()
    test_loss = 0
    accuracy = 0
    print("\nPERFORMING VALIDATION CHECK")
    for ii, (images, labels) in enumerate(validloader):
        images, labels = images.to(device), labels.to(device)
  
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

# Prompts a message and asks for a simple yes or no input
def get_yn_input(message):
    flag = False
    while (flag == False):
        answer = input('\n' + message + ' Please enter either Y or N\n')
        if(answer == 'Y' or answer == 'y'):
            flag = True
            return True
        elif(answer == 'N' or answer == 'n'):
            flag = True
            return False

# Prompts a message requesting for a directory path, checks if the path is a valid path else requests again
def get_dir_path(message):
    flag = False
    while (flag == False):
        dir_path = input('\n' + message + '\n')
        if(os.path.isdir(dir_path)):
            print('\nDir path is {}'.format(dir_path))
            flag = True
            return dir_path
        else:
            print('\nInvalid path\n')

# Prompts a message requesting for a file path, checks if the path is a valid path else requests again
def get_file_path(message):
    flag = False
    while (flag == False):
        file_path = input('\n' + message + '\n')
        if(os.path.isfile(file_path)):
            flag = True
            return file_path
        else:
            print('\nInvalid path\n')

# Prompts a message requesting for an int, checks if the type is an int else requests again
def get_int(message):
    flag = False
    input_int = None
    while (flag == False):
        input_int = input('\n' + message + '\n')
        try:
            input_int = int(input_int)
            flag = True
        except ValueError:
            print('\nPlease enter numbers only')
            flag = False
    return(input_int)

# Prompts a message requesting for an action, checks if the action is valid else requests again
def build_or_load_model():
    actions = ['build','load']
    flag = False
    while (flag != True):
        action = input("\nWould you like to build or load the neural network. Responses include: \n{}\n".format(actions))
        if(action == actions[0]):
            flag = True
            model = build_new_network()
        elif(action == actions[1]):
            flag = True
            model = load_checkpoint()
        else:
            print('\nInvalid action \n')
            
    return(model)

# Checks if GPU is available else runs on CPU
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('\nSelected device = {}\n'.format(device))
    return device

# Returns the model with the device loaded
def load_device(model):
    return model.to(get_device())

# Applies transformations to a single image
def process_image(image_path):

    print("\nPre-processing of image with path {}".format(image_path))

    image_pil = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])
    
    image_tensor = preprocess(image_pil)
    
    # unsqueeze to add a dimension for the batch. _ makes things happen in place
    image_tensor.unsqueeze_(0)
    
    return image_tensor

# Get the index to class JSON file
def get_idx_to_class():
    file_path = get_file_path('\nPlease enter the path of the index to class JSON file\n')
    with open(file_path, 'r') as f:
        return json.load(f)  
    
# Builds a new network
def build_new_network():
    
    # Gets a valid model before proceeding
    partial_model = helper.get_model(None)
        
    # Gets correct learn rate and hidden layers before proceeding
    learn_rate, hidden_layers = helper.get_network_inputs()

    # Buils the network
    model, criterion, optimizer = helper.build_network(learn_rate, hidden_layers, partial_model)
    model.criterion = criterion
    model.optimizer = optimizer

    return model
