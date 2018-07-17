import helper
import torch
import os
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import constants as Constant
from torchvision import datasets, transforms, models

# Trains the neural network
def train_network(model):

    optimizer = model.optimizer
    criterion = model.criterion

    valid_pass = False
    valid_loader = None
    train_loader = None
  
    epochs = helper.get_int('Please enter the epochs for training') 
    train_data_dir = helper.get_dir_path('Please enter path to the training data from current location')
    train_loader = helper.get_dataloader(train_data_dir, Constant.TRAIN)
    valid_pass = helper.get_yn_input('Do you want to do a validation pass?')

    if(valid_pass):
        valid_data_dir = helper.get_dir_path('Please enter path to the validating data from current location')
        valid_loader = helper.get_dataloader(valid_data_dir, Constant.VALID)


    print_every = len(train_loader)
    steps = 0
       
    print('\nTraining the network\n')

    device = helper.get_device()

    stepsArr = []
    accuracyArr = []

    print("\nTrainloader has {} images\n".format(len(train_loader)))

    # run a pre-set amount of times (epochs)
    for e in range(epochs):
        running_loss = 0
        accuracy = 0
    
        for images, labels in enumerate(train_loader):
            steps += 1
        
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            ps = torch.exp(outputs)
  
            #_, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item()
        
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy = equality.type(torch.FloatTensor).mean()    
            accString = "{:.2f}".format(accuracy*100)
            print("Step {} Test Loss {:.3f} Running Loss {:.3f} Accuracy {}%".format(steps,loss,running_loss,accString))
     
            stepsArr.append(steps)
            accuracyArr.append(accString)
      
            if (steps % print_every == 0 ):
                print("\n\nFinished Epoch: {}/{}.. ".format(e+1, epochs))

                if(valid_pass):
                    print("\nStarting validation pass for epoch: {}/{}.. ".format(e+1, epochs))
                    # Make sure network is in eval mode for inference
                    model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = helper.validation(model, valid_loader, criterion)

                    print("\nFinished validating for epoch: {}/{}.. ".format(e+1, epochs),
                            "Training loss: {:.3f}.. ".format(running_loss/print_every),
                            "Validation loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                            "Validation accuracy: {:.3f} %".format(accuracy/len(valid_loader)*100))

                    running_loss = 0

                    if(e == epochs):
                        break
                    else:
                        # Make sure training is back on
                        model.train()

    print("\n\nTraining finished")





if __name__ == "__main__":
    print("------------------------------ Main ------------------------------------")
else:
    print("------------------------------- Not Main -------------------------------")
    
