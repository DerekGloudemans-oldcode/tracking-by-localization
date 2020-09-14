# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:45:48 2020

@author: derek
"""
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
from _localizers.detrac_resnet34_localizer import ResNet34_Tracktor_Localizer2 as Localizer
from config.data_paths import data_paths

# surpress XML warnings
import warnings
warnings.filterwarnings(action='once')






def train_model(model, optimizer, scheduler,losses,
                    dataloaders,device, patience= 10, start_epoch = 0,
                    all_metrics = None, width_loss = None):
        """
        Alternates between a training step and a validation step at each epoch. 
        Validation results are reported but don't impact model weights
        """
        max_epochs = 20
        
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
                total_pre_acc = 0
                for inputs, targets,prevs in dataloaders[phase]:
                    
                    imsize = 224
                    wer = 3
                    
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    prevs = prevs.to(device).float()
                    reg_prevs = (prevs[:,:4]+imsize*(wer-1)/2)/(imsize*wer)
                    reg_prevs = reg_prevs.float()
                    
                    prev_widths = torch.zeros([reg_prevs.shape[0],2])
                    prev_widths[:,0] = reg_prevs[:,2] - reg_prevs[:,0] 
                    prev_widths[:,1] = reg_prevs[:,3] - reg_prevs[:,1]
                    prev_widths = prev_widths.float().to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        
                        try:
                            cls_out,reg_out = model(inputs,prev_widths)
                        
                            each_loss = []
                            # apply each reg loss function
                            # normalize targets
                            
                            reg_targets = (targets[:,:4]+imsize*(wer-1)/2)/(imsize*wer)
                            
                            reg_targets = reg_targets.float()
                            #reg_prevs = reg_prevs.float()
                            reg_out = reg_out.float()
                            
                            # compute offset relative to first frame bbox
                            #reg_out = reg_out + reg_prevs
                            
                            for loss_fn in losses['reg']:
                                loss_comp = loss_fn(reg_out,reg_targets) 
                                if phase == 'train':
                                    loss_comp.backward(retain_graph = True)
                                each_loss.append(round(loss_comp.item()*10000)/10000.0)
                            
                            # # # backprop loss from offset from previous bbox
                            # offset_loss = losses["reg"][0](reg_out,reg_prevs)/10.0
                            # if phase == 'train':
                            #     offset_loss.backward(retain_graph = True)
                            # each_loss.append(round(offset_loss.item()*10000)/10000.0)
                              
                            # backprop width loss here
                            # loss_comp = width_loss(reg_out,reg_prevs) * 10
                            # if phase == 'train':
                            #     loss_comp.backward(retain_graph = True)
                            # each_loss.append(round(loss_comp.item()*10000)/10000.0)  
                            
                            # apply each cls loss function
                            cls_targets = targets[:,4]
                            for loss_fn in losses['cls']:
                                loss_comp = loss_fn(cls_out.float(),cls_targets.long()) /10.0
                                if phase == 'train':
                                    loss_comp.backward()
                                each_loss.append(round(loss_comp.item()*10000)/10000.0)
                           
                            
                            # backpropogate loss and adjust model weights
                            if phase == 'train':
                                optimizer.step()
            
                            # compute IoU accuracy
                            acc,pre_acc = accuracy(reg_targets,reg_out,reg_prevs)
                        
                            count += 1
                            total_acc += acc
                            total_pre_acc += pre_acc
                            total_loss += sum(each_loss)
                            
                            del reg_out,reg_prevs,reg_targets,prevs,targets,inputs
                            
                        except RuntimeError:
                            print("Some sort of autograd error")
                            
                            for p in model.parameters():
                                if p.grad is not None:
                                    del p.grad
                            del prevs,targets,inputs  
                            
                            torch.cuda.empty_cache()
                            
                        
                    if count % 10 == 0:
                        print("{} epoch {} batch {} -- Loss so far: {:03f} -- {}".format(phase,epoch,count,total_loss/count,[item for item in each_loss]))
                        print("Pre-localization acc: {}  Post-localization acc: {}".format(total_pre_acc/count,total_acc/count))
                    #if count % 1000 == 0:
                        #plot_batch(model,next(iter(dataloaders['train'])),class_dict)
                    
                    
                    if count % 2000 == 0:
                        # save a checkpoint
                        PATH = "detrac_resnet34_tracktor_epoch{}_{}.pt".format(epoch,count)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 
                            "metrics": all_metrics
                            }, PATH)
                      
                    if count > 8000:
                        break
                    
                # report and record metrics at end of epoch
                avg_acc = total_acc/count
                avg_loss = total_loss/count
                print("Epoch {} avg {} loss: {:05f}  acc: {}".format(epoch, phase,avg_loss,avg_acc))
                all_metrics["{}_loss".format(phase)].append(total_loss)
                all_metrics["{}_acc".format(phase)].append(avg_acc)
                torch.cuda.empty_cache()
            
            if epoch % 1 == 0:
                plot_batch(model,next(iter(dataloaders['train'])),class_dict)

                if avg_loss < best_loss:
                    # save a checkpoint
                    PATH = "detrac_resnet34_tracktor_epoch{}.pt".format(epoch)
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


def accuracy(target,output,prev):
    """
    Computes iou accuracy vs target for both predictions and previous knowledge (input)
    """
    epsilon = 1e-06
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
    acc = iou.sum()/(len(iou)+epsilon)
    
    
    # repeat for prev
    minx,_ = torch.max(torch.cat((prev[:,0].unsqueeze(1),target[:,0].unsqueeze(1)),1),1)
    miny,_ = torch.max(torch.cat((prev[:,1].unsqueeze(1),target[:,1].unsqueeze(1)),1),1)
    maxx,_ = torch.min(torch.cat((prev[:,2].unsqueeze(1),target[:,2].unsqueeze(1)),1),1)
    maxy,_ = torch.min(torch.cat((prev[:,3].unsqueeze(1),target[:,3].unsqueeze(1)),1),1)
    
    zeros = torch.zeros(minx.shape).unsqueeze(1).to(device)
    delx,_ = torch.max(torch.cat(((maxx-minx).unsqueeze(1),zeros),1),1)
    dely,_ = torch.max(torch.cat(((maxy-miny).unsqueeze(1),zeros),1),1)
    intersection = torch.mul(delx,dely)
    a1 = torch.mul(prev[:,2]-prev[:,0],prev[:,3]-prev[:,1])
    a2 = torch.mul(target[:,2]-target[:,0],target[:,3]-target[:,1])
    #a1,_ = torch.max(torch.cat((a1.unsqueeze(1),zeros),1),1)
    #a2,_ = torch.max(torch.cat((a2.unsqueeze(1),zeros),1),1)
    union = a1 + a2 - intersection 
    iou = intersection / (union + epsilon)
    pre_acc = iou.sum() / (len(iou) + epsilon)
    
    return acc, pre_acc


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
    
    def forward(self,output,prev):
        """ Compute MSE between output width and previous width"""
        
        out_width   = output[:,2] - output[:,0]
        out_height  = output[:,3] - output[:,1]
        prev_width  =   prev[:,2] -   prev[:,0]
        prev_height =   prev[:,3] -   prev[:,1]
        
        diff_width  = out_width  - prev_width
        diff_height = out_height - prev_height
        diff = torch.cat((diff_width,diff_height),dim = 0)
        square = torch.pow(diff,2)
        
        loss = square.sum() / len(square)
        
        return loss
    
        
def plot_batch(model,batch,class_dict):
    """
    Given a batch and corresponding labels, plots both model predictions
    and ground-truth
    model - Localize_Net() object
    batch - batch from loader loading Detrac_Localize_Dataset() data
    class-dict - dict for converting between integer and string class labels
    """
    wer = 3
    imsize = 224
        
    input = batch[0]
    label = batch[1]
    prev = batch[1]
    prev = prev.to(device).float()
    reg_prevs = (prev[:,:4]+imsize*(wer-1)/2)/(imsize*wer)
    reg_prevs = reg_prevs.float()
    
    reg_prevs[:,0] = reg_prevs[:,2] - reg_prevs[:,0]
    reg_prevs[:,1] = reg_prevs[:,3] - reg_prevs[:,1]
    reg_prevs = reg_prevs[:,:2]
    input = input.to(device)
    
    cls_label = label[:,4]
    reg_label = label[:,:4]
    cls_output, reg_output = model(input,reg_prevs)
    
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
    
    new_checkpoint = "cpu" + checkpoint 
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': new_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        "metrics": all_metrics
        }, new_checkpoint)

def old_checkpoint(model,cp):
    cp = torch.load(cp)
    cp = cp["model_state_dict"]
    cp["regressor2.0.bias"] = cp["regressor.2.bias"].cpu()     # size 4
    cp["regressor2.0.weight"] = cp["regressor.2.weight"].cpu() # size 4 x 31
    del cp["regressor.2.bias"],cp["regressor.2.weight"]
    
    # append 4 new values to regressor2.0.weight
    new = torch.randn([4,4])*0.2 
    cp["regressor2.0.weight"] = torch.cat((cp["regressor2.0.weight"],new), axis = 1)
    
    delete = []
    for key in cp.keys():
        if "embedder" in key:
            delete.append(key)
    
    delete.reverse()
    
    for key in delete:
        del cp[key]
        
    model.load_state_dict(cp)# = cp
    
    return model

def tracktor1_2_convert(model,cp):
     weighting = 0.2
     cp = torch.load(cp)
     cp = cp["model_state_dict"]
     cp["regressor2.0.weight"] = torch.cat((cp["regressor2.0.weight"], torch.randn([31,35]).to(device)*weighting),axis = 0)
     cp["regressor2.0.bias"] = torch.cat((cp["regressor2.0.bias"], torch.randn([31]).to(device)*weighting),axis = 0)
     cp["regressor2.2.weight"] = torch.randn([4,35]).to(device) * weighting
     cp["regressor2.2.weight"][:4,:4] = torch.eye(4).to(device)
     cp["regressor2.2.bias"] = torch.randn(4).to(device) * weighting
     
     model.load_state_dict(cp)
     return model
    

#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    
    checkpoint_file = "TRACKTOR2_INIT.pt"
    checkpoint_file = "detrac_resnet34_tracktor_epoch7_6000.pt"
    #checkpoint_file = "TRACKTOR_SAVE.pt"#
    #checkpoint_file = None
    patience = 1
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
    # cp = "cpu_detrac_resnet34_alpha.pt"
    # model = old_checkpoint(model,cp)    
    
    # model = tracktor1_2_convert(model,checkpoint_file)
    #checkpoint_file = None
    
    # 3. create training params
    params = {'batch_size' : 24,
              'shuffle'    : True,
              'num_workers': 0 ,
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
    
    # group dataloaders 
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    print("Got dataloaders. {},{}".format(datasizes['train'],datasizes['val']))
    
    # 5. define stochastic gradient descent optimizer    
    #optimizer = optim.SGD(model.parameters(), lr = lr_init, momentum = 0.3)
    
    #freeze all but last layer
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.regressor2.parameters():
    #     param.requires_grad = True
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
      
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr_init, momentum = 0.9)    
    
    # 7. define start epoch for consistent labeling if checkpoint is 
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = True, mode = "min", patience = patience, factor=0.3)
    start_epoch = -1
    all_metrics = None


    if MULTI:
        model = nn.DataParallel(model,device_ids = [0,1])
        model = model.to(device)
        print("Loaded model.")  


    # 8. if checkpoint specified, load model and optimizer weights from checkpoint
    if checkpoint_file != None:
        model,_,start_epoch,all_metrics = load_model(checkpoint_file, model, optimizer)
        #model,_,start_epoch = load_model(checkpoint_file, model, optimizer) # optimizer restarts from scratch
        print("Checkpoint loaded.")

        
    # 9. define losses
    losses = {"cls": [nn.CrossEntropyLoss()],
              "reg": [nn.MSELoss(),Box_Loss(),]
              }
    
    # losses = {"cls": [],
    #           "reg": [Box_Loss()]
    #           }
    
    #optimizer = optim.Adam(model.parameters(), lr=lr_init)

    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr_init, momentum = 0.3)    

    #freeze all but last layer
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.module.regressor2.parameters():
    #     param.requires_grad = True
    # for param in model.module.classifier.parameters():
    #     param.requires_grad = True
      
    
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr_init, momentum = 0.9)    

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
                            all_metrics = all_metrics,
                            width_loss = Width_Loss())
        
   