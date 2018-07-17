import helper
import torch
import os
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import constants as Constant
from torchvision import datasets, transforms, models

# Predicts the class of an image using a trained model
def predict(model):
    
    if(model == None):
        model = helper.load_checkpoint()

    image_path = helper.get_file_path('\nPlease enter the path of the image you want to analyse\n')
    topk = helper.get_int('\nPlease enter how many to the top predictions you want to see (topk)\n')

    device = helper.get_device()
    model = helper.load_device(model)

    image_tensor = helper.process_image(image_path).to(device)
    idx_to_class = helper.get_idx_to_class()
    print('\nPredicting\n')

    with torch.no_grad():
        output = model.forward(image_tensor)
    
    ps = torch.exp(output)
    
    topK_ps = torch.topk(ps, topk)
    
    probs = topK_ps[0].cpu().numpy().squeeze()
    sorted_ps_label_keys = topK_ps[1].cpu().numpy().squeeze()
    get_label = lambda x:idx_to_class[str(x)]
    
    classes = []
        
    for i in sorted_ps_label_keys[0:topk]:
        classes.append(get_label(i))
    
    print('\nFinished predicting\n')
    return probs, classes

# Handles for initiating from predict.py
if __name__ == "__main__":
    predict(None)
