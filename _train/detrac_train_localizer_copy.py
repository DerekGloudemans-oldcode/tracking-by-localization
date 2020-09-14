# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:45:48 2020

@author: d"""
import os
import sys,inspect
import numpy as np
import random 
import math
import time
random.seed = 0

import cv2
from PIL import Image
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms,models
from torchvision.transforms import functional as F
import matplotlib.pyplot  as plt



# add all packages and directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parent_dir)

from config.data_paths import directories
for item in directories:
    sys.path.insert(0,item)

from _data_utils.detrac.detrac_localization_dataset import Localize_Dataset, class_dict
from _localizers.detrac_resnet34_localizer import ResNet34_Tracktor_Localizer as Localizer
from config.data_paths import data_paths

# surpress XML warnings
import warnings
warnings.filterwarnings(action='once')






def train_model(model, optimizer, scheduler,losses,
                    dataloaders,device, patience= 10, start_epoch = 0,
                    all_metrics = None):
        """
        Alternates between a training step and a validation step at each epoch. 
        Validation results are reported but don't impact model weights
        """
        max_epochs = 200
        
        # for storing all metrics
        if all_metrics == None:
          all_metrics = {
                  'train_loss':[],
                  'val_loss':[],
                  "train_acc":[],
                  "val_acc":[]
                  }
        
        # for early stopping
        best_loss = np.inf
        epochs_since_improvement = 0

        for epoch in range(start_epoch,max_epochs):
            for phase in ["train","val"]:
                if phase == 'train':
                    scheduler.step(best_loss) # adjust learning rate if plateauing
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                # Iterate over data
                count = 0
                total_loss = 0
                total_acc = 0
                for inputs, targets,_ in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        cls_out,reg_out = model(inputs)
                        
                        each_loss = []
                        # apply each reg loss function
                        # normalize targets
                        imsize = 224
                        wer = 3
                        reg_targets = (targets[:,:4]+imsize*(wer-1)/2)/(imsize*wer)
                        acc = 0
                        
                        try:
                            for loss_fn in losses['reg']:
                                loss_comp = loss_fn(reg_out.float(),reg_targets.float()) 
                                if phase == 'train':
                                    loss_comp.backward(retain_graph = True)
                                each_loss.append(round(loss_comp.item()*100000)/100000.0)
                                
                            # apply each cls loss function
                            cls_targets = targets[:,4]
                            for loss_fn in losses['cls']:
                                loss_comp = loss_fn(cls_out.float(),cls_targets.long()) /100.0
                                if phase == 'train':
                                    loss_comp.backward()
                                each_loss.append(round(loss_comp.item()*100000)/100000.0)
                           
                            
                            # backpropogate loss and adjust model weights
                            if phase == 'train':
                                optimizer.step()
            
                        except RuntimeError:
                            print("Some sort of autograd error")
                            
                    # verbose update
                    count += 1
                    total_acc += acc
                    total_loss += sum(each_loss) #loss.item()
                    if count % 10 == 0:
                        print("{} epoch {} batch {} -- Loss so far: {:03f} -- {}".format(phase,epoch,count,total_loss/count,[item for item in each_loss]))
                   
                    # if count % 1000 == 0:
                    #     plot_batch(model,next(iter(dataloaders['train'])),class_dict)
                    
                            
                # report and record metrics at end of epoch
                avg_acc = total_acc/count
                avg_loss = total_loss/count
                print("Epoch {} avg {} loss: {:05f}  acc: {}".format(epoch, phase,avg_loss,avg_acc))
                all_metrics["{}_loss".format(phase)].append(total_loss)
                all_metrics["{}_acc".format(phase)].append(avg_acc)
            
            if epoch % 1 == 0:
                plot_batch(model,next(iter(dataloaders['train'])),class_dict)

                if avg_loss < best_loss:
                    # save a checkpoint
                    PATH = "detrac_resnet34_epoch{}.pt".format(epoch)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "metrics": all_metrics
                        }, PATH)
                
                torch.cuda.empty_cache()
                
            # stop training when there is no further improvement
            if avg_loss < best_loss:
                epochs_since_improvement = 0
                best_loss = avg_loss
            else:
                epochs_since_improvement +=1
            
            print("{} epochs since last improvement.".format(epochs_since_improvement))
            if epochs_since_improvement >= patience:
                break
                
        return model , all_metrics

def load_model(checkpoint_file,model,optimizer):
    """
    Reloads a checkpoint, loading the model and optimizer state_dicts and 
    setting the start epoch
    """
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    all_metrics = checkpoint['metrics']
    
    return model,optimizer,epoch,all_metrics

class Box_Loss(nn.Module):        
    def __init__(self):
        super(Box_Loss,self).__init__()
        
    def forward(self,output,target,epsilon = 1e-07):
        """ Compute the bbox iou loss for target vs output using tensors to preserve
        gradients for efficient backpropogation"""
        
        # minx miny maxx maxy
        minx,_ = torch.max(torch.cat((output[:,0].unsqueeze(1),target[:,0].unsqueeze(1)),1),1)
        miny,_ = torch.max(torch.cat((output[:,1].unsqueeze(1),target[:,1].unsqueeze(1)),1),1)
        maxx,_ = torch.min(torch.cat((output[:,2].unsqueeze(1),target[:,2].unsqueeze(1)),1),1)
        maxy,_ = torch.min(torch.cat((output[:,3].unsqueeze(1),target[:,3].unsqueeze(1)),1),1)
        
        zeros = torch.zeros(minx.shape).unsqueeze(1).to(device)
        delx,_ = torch.max(torch.cat(((maxx-minx).unsqueeze(1),zeros),1),1)
        dely,_ = torch.max(torch.cat(((maxy-miny).unsqueeze(1),zeros),1),1)
        intersection = torch.mul(delx,dely)
        a1 = torch.mul(output[:,2]-output[:,0],output[:,3]-output[:,1])
        a2 = torch.mul(target[:,2]-target[:,0],target[:,3]-target[:,1])
        #a1,_ = torch.max(torch.cat((a1.unsqueeze(1),zeros),1),1)
        #a2,_ = torch.max(torch.cat((a2.unsqueeze(1),zeros),1),1)
        union = a1 + a2 - intersection 
        iou = intersection / (union + epsilon)
        #iou = torch.clamp(iou,0)
        return 1- iou.sum()/(len(iou)+epsilon)
    
class Width_Loss(nn.Module):        
    def __init__(self):
        super(Width_Loss,self).__init__()
        
    def forward(self,output,target,epsilon = 1e-07):
        """ Compute the bbox iou loss for target vs output using tensors to preserve
        gradients for efficient backpropogation"""
        
        # minx miny maxx maxy
        pred_width = output[:,2]-output[:,0]
        targ_width = target[:,2]-target[:,0]
        error = torch.abs(pred_width - targ_width) / targ_width
        loss  = error.mean()

        return loss
  
def plot_batch(model,batch,class_dict):
    """
    Given a batch and corresponding labels, plots both model predictions
    and ground-truth
    model - Localize_Net() object
    batch - batch from loader loading Detrac_Localize_Dataset() data
    class-dict - dict for converting between integer and string class labels
    """
    input = batch[0]
    label = batch[1]
    cls_label = label[:,4]
    reg_label = label[:,:4]
    cls_output, reg_output = model(input)
    
    _,cls_preds = torch.max(cls_output,1)
    batch = input.data.cpu().numpy()
    bboxes = reg_output.data.cpu().numpy()
    
    # define figure subplot grid
    batch_size = len(cls_label)
    row_size = min(batch_size,8)
    fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True)
    
    # for image in batch, put image and associated label in grid
    for i in range(0,batch_size):
        
        # get image
        im   = batch[i].transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        im   = std * im + mean
        im   = np.clip(im, 0, 1)
        
        # get predictions
        cls_pred = cls_preds[i].item()
        bbox = bboxes[i]
        
        # get ground truths
        cls_true = cls_label[i].item()
        reg_true = reg_label[i]
        
        wer = 3
        imsize = 224
        
        # convert to normalized coords
        reg_true = (reg_true+imsize*(wer-1)/2)/(imsize*wer)
        # convert to im coords
        reg_true = (reg_true* 224*wer - 224*(wer-1)/2).int()
        
        # transform bbox coords back into im pixel coords
        bbox = (bbox* 224*wer - 224*(wer-1)/2).astype(int)
        # plot pred bbox
        im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.1,0.6,0.9),2)
       
        # plot ground truth bbox
        im = cv2.rectangle(im,(reg_true[0],reg_true[1]),(reg_true[2],reg_true[3]),(0.6,0.1,0.9),2)
        im = im.get()
                
        # title with class preds and gt
        label = "{} -> ({})".format(class_dict[cls_pred],class_dict[cls_true])
        if batch_size <= 8:
            axs[i].imshow(im)
            axs[i].set_title(label)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i//row_size,i%row_size].imshow(im)
            axs[i//row_size,i%row_size].set_title(label)
            axs[i//row_size,i%row_size].set_xticks([])
            axs[i//row_size,i%row_size].set_yticks([])
        plt.pause(.001)    
    plt.close()

def move_dual_checkpoint_to_cpu(model,optimizer,checkpoint):
    model,optimizer,epoch,all_metrics = load_model(checkpoint, model, optimizer)
    model = nn.DataParallel(model,device_ids = [0])
    model = model.to(device)
    
    new_state_dict = {}
    for key in model.state_dict():
        new_state_dict[key.split("module.")[-1]] = model.state_dict()[key]
    
    new_checkpoint = checkpoint.split("resnet")[0] + "cpu_resnet18_epoch{}.pt".format(epoch)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': new_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        "metrics": all_metrics
        }, new_checkpoint)
    

#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    
    checkpoint_file = None
    
    patience = 3
    lr_init = 0.001
    
    label_dir       = data_paths["train_lab"]
    train_image_dir = data_paths["train_partition"]
    test_image_dir  = data_paths["val_partition"]
    
    
    #################### Change nothing below here #######################
    
    # 1. CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        MULTI = True
    else:
        MULTI = False
    torch.cuda.empty_cache()   
    
    # 2. load model
    model = Localizer()
    cp = "TRACTOR_INIT.pt"
    cp = torch.load(cp)
    model.load_state_dict(cp)

    if MULTI:
        model = nn.DataParallel(model,device_ids = [0,1])
    model = model.to(device)
    print("Loaded model.")
    
    
    # 3. create training params
    params = {'batch_size' : 64,
              'shuffle'    : True,
              'num_workers':0,
              'drop_last'  : True
              }
    
    # 4. create dataloaders
    try:   
        len(train_data)
        len(test_data)
    except:   
        train_data = Localize_Dataset(train_image_dir, label_dir)
        test_data =  Localize_Dataset(test_image_dir,label_dir)
        
    trainloader = data.DataLoader(train_data, **params)
    testloader = data.DataLoader(test_data, **params)
    
    # group dataloaders #cp = "detrac_resnet34_alpha.pt"
    #cp = torch.load(cp)
    #model.load_state_dict(cp['model_state_dict'])
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    print("Got dataloaders. {},{}".format(datasizes['train'],datasizes['val']))
    
    # 5. define stochastic gradient descent optimizer    
    #optimizer = optim.Adam(model.parameters(), lr=lr_init)
    optimizer = optim.SGD(model.parameters(), lr = lr_init, momentum = 0.3)
    # 6. decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = True, mode = "min", patience = patience, factor=0.3)
    
    # 7. define start epoch for consistent labeling if checkpoint is reloaded
    start_epoch = -1
    all_metrics = None

    # 8. if checkpoint specified, load model and optimizer weights from checkpoint
    if checkpoint_file != None:
        model,_,start_epoch,all_metrics = load_model(checkpoint_file, model, optimizer)
        #model,_,start_epoch = load_model(checkpoint_file, model, optimizer) # optimizer restarts from scratch
        print("Checkpoint loaded.")
     
    # 9. define losses
    losses = {"cls": [nn.CrossEntropyLoss()],
              "reg": [Width_Loss(), Box_Loss(),]
              }
    # losses = {"cls": [],
    #             "reg": [nn.MSELoss,Box_Loss()]
    #             }
    
    if True:    
    # train model
        print("Beginning training.")
        model,all_metrics = train_model(model,
                            optimizer, 
                            exp_lr_scheduler,
                            losses,
                            dataloaders,
                            device,
                            patience = patience*2,
                            start_epoch = start_epoch+1,
                            all_metrics = all_metrics)
        
   