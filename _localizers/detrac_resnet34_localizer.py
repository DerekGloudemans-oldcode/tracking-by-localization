# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:45:48 2020
@author: derek
"""

import numpy as np
import torch
from torch import nn
from torchvision import models

# define ResNet18 based network structure
class ResNet34_Localizer(nn.Module):
    
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate some nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet34_Localizer, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        #self.feat = models.resnet18(pretrained=True)
        self.feat = models.resnet34(pretrained = True)
        # get size of some layers
        start_num = self.feat.fc.out_features
        mid_num = int(np.sqrt(start_num))
        
        cls_out_num = 13
        reg_out_num = 4 # bounding box coords
        embed_out_num = 128
        
        # define classifier
        self.classifier = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,cls_out_num,bias = True)
                          #nn.Softmax(dim = 1)
                          )
        
        # define regressor
        self.regressor = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,reg_out_num,bias = True),
                          nn.ReLU()
                          )
        
        self.embedder = nn.Sequential(
                  nn.Linear(start_num,start_num // 3,bias=True),
                  nn.ReLU(),
                  nn.Linear(start_num // 3,embed_out_num,bias = True),
                  nn.ReLU()
                  )
        
        for layer in self.classifier:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
        for layer in self.regressor:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
        for layer in self.embedder:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
            
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        features = self.feat(x)
        cls_out = self.classifier(features)
        reg_out = self.regressor(features)
        #embed_out = self.embedder(features)
        #out = torch.cat((cls_out, reg_out), 0) # might be the wrong dimension
        
        return cls_out,reg_out


class ResNet34_Tracktor_Localizer(nn.Module):
    
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate some nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet34_Tracktor_Localizer, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        #self.feat = models.resnet18(pretrained=True)
        self.feat = models.resnet34(pretrained = True)
        # get size of some layers
        start_num = self.feat.fc.out_features
        mid_num = int(np.sqrt(start_num))
        
        cls_out_num = 13
        reg_out_num = 4 # bounding box coords
        embed_out_num = 128
        
        # define classifier
        self.classifier = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,cls_out_num,bias = True)
                          #nn.Softmax(dim = 1)
                          )
        
        # define regressor
        self.regressor = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU())
        self.regressor2 = nn.Sequential(
                          nn.Linear(mid_num+4,reg_out_num,bias = True),
                          nn.ReLU()
                          )
        
        # self.embedder = nn.Sequential(
        #           nn.Linear(start_num,start_num // 3,bias=True),
        #           nn.ReLU(),
        #           nn.Linear(start_num // 3,embed_out_num,bias = True),
        #           nn.ReLU()
        #           )
        
        for layer in self.classifier:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
        for layer in self.regressor:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
        # for layer in self.embedder:
        #     if type(layer) == torch.nn.modules.linear.Linear:
        #         init_val = 0.05
        #         nn.init.uniform_(layer.weight.data,-init_val,init_val)
        #         nn.init.uniform_(layer.bias.data,-init_val,init_val)
            
    def forward(self, x,prev_x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        features = self.feat(x)
        cls_out = self.classifier(features)
        inter_in = self.regressor(features)
        inter_out = torch.cat((inter_in,prev_x),dim = 1)
        reg_out = self.regressor2(inter_out)
        
        #embed_out = self.embedder(features)
        #out = torch.cat((cls_out, reg_out), 0) # might be the wrong dimension
        
        return cls_out,reg_out
    
class ResNet34_Tracktor_Localizer2(nn.Module):
    
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate some nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet34_Tracktor_Localizer2, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        #self.feat = models.resnet18(pretrained=True)
        self.feat = models.resnet34(pretrained = True)
        # get size of some layers
        start_num = self.feat.fc.out_features
        mid_num = int(np.sqrt(start_num))
        
        cls_out_num = 13
        reg_out_num = 4 # bounding box coords
        embed_out_num = 128
        
        # define classifier
        self.classifier = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,cls_out_num,bias = True)
                          #nn.Softmax(dim = 1)
                          )
        
        # define regressor
        self.regressor = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU())
        self.regressor2 = nn.Sequential(
                          #nn.Dropout(0.1),
                          nn.Linear(mid_num+2,reg_out_num,bias = True),
                          nn.ReLU()
                          )
        
        # self.embedder = nn.Sequential(
        #           nn.Linear(start_num,start_num // 3,bias=True),
        #           nn.ReLU(),
        #           nn.Linear(start_num // 3,embed_out_num,bias = True),
        #           nn.ReLU()
        #           )
        
        for layer in self.classifier:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
        for layer in self.regressor:
            if type(layer) == torch.nn.modules.linear.Linear:
                init_val = 0.05
                nn.init.uniform_(layer.weight.data,-init_val,init_val)
                nn.init.uniform_(layer.bias.data,-init_val,init_val)
        # for layer in self.embedder:
        #     if type(layer) == torch.nn.modules.linear.Linear:
        #         init_val = 0.05
        #         nn.init.uniform_(layer.weight.data,-init_val,init_val)
        #         nn.init.uniform_(layer.bias.data,-init_val,init_val)
            
    def forward(self, x,prev_x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        features = self.feat(x)
        cls_out = self.classifier(features)
        inter_in = self.regressor(features)
        inter_out = torch.cat((inter_in,prev_x),dim = 1)
        reg_out = self.regressor2(inter_out)
        
        #embed_out = self.embedder(features)
        #out = torch.cat((cls_out, reg_out), 0) # might be the wrong dimension
        
        return cls_out,reg_out

