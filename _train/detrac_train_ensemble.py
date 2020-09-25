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
import _pickle as pickle
random.seed = 0

import cv2
from PIL import Image
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms,models
from torchvision.transforms import functional as F
import matplotlib.pyplot  as plt
from torchvision.ops import roi_align


# add all packages and directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parent_dir)

from config.data_paths import directories
for item in directories:
    sys.path.insert(0,item)

from _data_utils.detrac.detrac_localization_dataset import class_dict
from _data_utils.detrac.detrac_tracking_dataset import Track_Dataset
from _localizers.detrac_resnet34_localizer import ResNet34_Conf_Localizer2
from config.data_paths import data_paths
from detrac_train_localizer_conf import Box_Loss, Conf_Loss, move_dual_checkpoint_to_cpu
from detrac_train_tracktor_localizer import accuracy
from _tracker.torch_kf_dual import Torch_KF

# surpress XML warnings
import warnings
warnings.filterwarnings(action='once')

def init_filters(states,ims,model,kf_params,n = 2):
    
    tracker = Torch_KF("cpu",INIT = kf_params, ADD_MEAN_Q = True, ADD_MEAN_R = True)
    obj_ids = [i for i in range(len(states))] 

    # possibly add noise to states here        
    tracker.add(states[:,0,:4],obj_ids)
    
    for frame_idx in range(1,n+1):
     
             # get a priori error
             tracker.predict()
             objs = tracker.objs()
             objs = [objs[key] for key in objs]
             a_priori = torch.from_numpy(np.array(objs)).double()
            
             prev = tracker.X[:,:4]
             # prevs = objs
             # prevs = [torch.from_numpy(item)[:4] for item in prevs]
             # prev = torch.stack(prevs)
             
             prevs = torch.zeros(prev.shape)
             prevs[:,0] = prev[:,0] - prev[:,2]/2.0
             prevs[:,2] = prev[:,0] + prev[:,2]/2.0
             prevs[:,1] = prev[:,1] - prev[:,2]*prev[:,3]/2.0
             prevs[:,3] = prev[:,1] + prev[:,2]*prev[:,3]/2.0 
            
             # ims are collated by frame,then batch index
             relevant_ims = ims[frame_idx]
             frames =[]
             for item in relevant_ims:
                 with Image.open(item) as im:
                        im = F.to_tensor(im)
                        frame = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                        frames.append(frame)
             frames = torch.stack(frames).to(device)
             
             # crop image
             boxes = a_priori[:,:4]
             
             # convert xysr boxes into xmin xmax ymin ymax
             # first row of zeros is batch index (batch is size 0) for ROI align
             new_boxes = np.zeros([len(boxes),5]) 
     
             # use either s or s x r for both dimensions, whichever is larger,so crop is square
             #box_scales = np.max(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1)
             box_scales = np.min(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
                 
             #expand box slightly
             ber = 2.15
             box_scales = box_scales * ber# box expansion ratio
             
             new_boxes[:,1] = boxes[:,0] - box_scales/2
             new_boxes[:,3] = boxes[:,0] + box_scales/2 
             new_boxes[:,2] = boxes[:,1] - box_scales/2 
             new_boxes[:,4] = boxes[:,1] + box_scales/2 
             for i in range(len(new_boxes)):
                 new_boxes[i,0] = i # set image index for each
                 
             torch_boxes = torch.from_numpy(new_boxes).float().to(device)
             
             # crop using roi align
             crops = roi_align(frames,torch_boxes,(224,224))
             
             new_prevs = torch.zeros(prevs.shape)
             new_prevs[:,0] = prevs[:,0] - new_boxes[:,1]
             new_prevs[:,1] = prevs[:,1] - new_boxes[:,2]
             new_prevs[:,2] = prevs[:,2] - new_boxes[:,1]
             new_prevs[:,3] = prevs[:,3] - new_boxes[:,2]
             new_prevs = new_prevs * 224/torch.from_numpy(box_scales).unsqueeze(1).repeat(1,4) 
             
             _,reg_out = model(crops,new_prevs.float())
             torch.cuda.synchronize()
     
             # 5b. convert to global image coordinates 
                 
             # these detections are relative to crops - convert to global image coords
             wer = 1.25
             detections = (reg_out* 224*wer - 224*(wer-1)/2)
             detections = detections.data.cpu()
             
             # add in original box offsets and scale outputs by original box scales
             detections[:,0] = detections[:,0]*box_scales/224 + new_boxes[:,1]
             detections[:,2] = detections[:,2]*box_scales/224 + new_boxes[:,1]
             detections[:,1] = detections[:,1]*box_scales/224 + new_boxes[:,2]
             detections[:,3] = detections[:,3]*box_scales/224 + new_boxes[:,2]
     
     
             # convert into xysr form 
             output = np.zeros([len(detections),4])
             output[:,0] = (detections[:,0] + detections[:,2]) / 2.0
             output[:,1] = (detections[:,1] + detections[:,3]) / 2.0
             output[:,2] = (detections[:,2] - detections[:,0])
             output[:,3] = (detections[:,3] - detections[:,1]) / output[:,2]
             pred = torch.from_numpy(output)
             
             tracker.update(pred,obj_ids)
             tracker.predict()
             
             objs = tracker.objs()
             objs = [objs[key] for key in objs]
             a_priori = torch.from_numpy(np.array(objs)).double() 
             
    return a_priori[:,:4]

def get_crops(prev,ims,states,device):
    prevs = torch.zeros(prev.shape)
    prevs[:,0] = prev[:,0] - prev[:,2]/2.0
    prevs[:,2] = prev[:,0] + prev[:,2]/2.0
    prevs[:,1] = prev[:,1] - prev[:,2]*prev[:,3]/2.0
    prevs[:,3] = prev[:,1] + prev[:,2]*prev[:,3]/2.0 
   
    gts = states[:,-1,:4]
    gtxyxy = torch.zeros(gts.shape)
    gtxyxy[:,0] = gts[:,0] - gts[:,2]/2.0
    gtxyxy[:,2] = gts[:,0] + gts[:,2]/2.0
    gtxyxy[:,1] = gts[:,1] - gts[:,2]*gts[:,3]/2.0
    gtxyxy[:,3] = gts[:,1] + gts[:,2]*gts[:,3]/2.0 
    
    # ims are collated by frame,then batch index
    relevant_ims = ims[-1]
    frames =[]
    for item in relevant_ims:
        with Image.open(item) as im:
               im = F.to_tensor(im)
               frame = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
               frames.append(frame)
    frames = torch.stack(frames).to(device)
    
    # crop image
    boxes = prev
    
    # convert xysr boxes into xmin xmax ymin ymax
    # first row of zeros is batch index (batch is size 0) for ROI align
    new_boxes = np.zeros([len(boxes),5]) 

    # use either s or s x r for both dimensions, whichever is larger,so crop is square
    #box_scales = np.max(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1)
    box_scales = np.min(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
        
    #expand box slightly
    ber = 2.0
    box_scales = box_scales * ber# box expansion ratio
    
    new_boxes[:,1] = boxes[:,0] - box_scales/2
    new_boxes[:,3] = boxes[:,0] + box_scales/2 
    new_boxes[:,2] = boxes[:,1] - box_scales/2 
    new_boxes[:,4] = boxes[:,1] + box_scales/2 
    for i in range(len(new_boxes)):
        new_boxes[i,0] = i # set image index for each
        
    torch_boxes = torch.from_numpy(new_boxes).float().to(device)
    
    # crop using roi align
    crops = roi_align(frames,torch_boxes,(224,224))
    crops = crops.to(device)
    
    new_prevs = torch.zeros(prevs.shape)
    new_prevs[:,0] = prevs[:,0] - new_boxes[:,1]
    new_prevs[:,1] = prevs[:,1] - new_boxes[:,2]
    new_prevs[:,2] = prevs[:,2] - new_boxes[:,1]
    new_prevs[:,3] = prevs[:,3] - new_boxes[:,2]
    new_prevs = new_prevs * 224/torch.from_numpy(box_scales).unsqueeze(1).repeat(1,4) 
    new_prevs = new_prevs.float().to(device)
    
    new_gts = torch.zeros(gtxyxy.shape)
    new_gts[:,0] = gtxyxy[:,0] - new_boxes[:,1]
    new_gts[:,1] = gtxyxy[:,1] - new_boxes[:,2]
    new_gts[:,2] = gtxyxy[:,2] - new_boxes[:,1]
    new_gts[:,3] = gtxyxy[:,3] - new_boxes[:,2]
    new_gts = new_gts * 224/torch.from_numpy(box_scales).unsqueeze(1).repeat(1,4) 
    new_gts = new_gts.float().to(device)
    
    return crops, new_prevs, new_gts

class Box_Loss(nn.Module):        
    def __init__(self):
        super(Box_Loss,self).__init__()
        
    def forward(self,output,target,device,epsilon = 1e-07):
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


def accuracy(target,output,prev,device):
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


def train(model,
          optimizer, 
          scheduler,
          losses,
          dataloaders,
          device,
          patience= 10,
          start_epoch = 0,
          kf_params = None
          ):
    """

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.
    scheduler : TYPE
        DESCRIPTION.
    losses : TYPE
        DESCRIPTION.
    dataloaders : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    patience : TYPE, optional
        DESCRIPTION. The default is 10.
    start_epoch : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    
    wait = 0 # how many epochs since last improvement
    best_acc = 0
    epoch = 0
    box_loss = Box_Loss()
    
    while wait < patience:
    
        #### fit R
    
        for phase in ["train","val"]:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data
            count = 0
            total_loss = 0
            total_acc = 0
            total_pre_acc = 0
            running_loss = []
            
            for batch in dataloaders[phase]:
                states = batch[0]
                im_paths = batch[1]
                
                
                #### initialize filters for batch
                with torch.no_grad():
                    prevs = init_filters(states,im_paths,model,kf_params,n = 2)
                    crops,reg_prev,reg_target = get_crops(prevs,im_paths,states,device)
                    
                #### parse prevs and format inputs and targets correctly 
                
                # zero the parameter gradients
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase == 'train'):
                        cls_out,reg_out = model(crops,reg_prev.float())
                
                #### apply loss function (box loss only here)
                loss = box_loss(reg_out,reg_target,device)
                loss.backward()
                
                # backpropogate loss and adjust model weights
                if phase == 'train':
                    optimizer.step()
                
                acc,pre_acc = accuracy(reg_out,reg_prev,reg_target,device)
                
                #### store metrics
                total_loss += loss.item()
                running_loss.append(loss.item)
                if len(running_loss) > 200:
                    del running_loss[0]
                total_acc += acc
                total_pre_acc += pre_acc
                count += 1
                
                del states,im_paths,batch,loss,reg_prev,reg_target,reg_out,cls_out
                
                if count % 1 == 0:
                    avg_acc = total_acc/count
                    avg_pre_acc = total_pre_acc/count
                    avg_loss = sum(running_loss)/len(running_loss)
                    print ("E{}B{}: loss: {},  pre acc: {},  post acc: {}".format(epoch,count,avg_acc,avg_pre_acc))
                # what was the previous IOU, and what was the post-localization IOU
                            
                        
    
#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    
    checkpoint_file = "detrac_resnet34_wer125_epoch6.pt"
    checkpoint_file = "CONF_INIT.pt"
    checkpoint_file = "ENSEMBLE_INIT.pt"
    filter_directory = data_paths['filter_params']
    kf_params_file =  os.path.join(filter_directory,"detrac_7_QR_conf.cpkl")
    
    patience = 3
    lr_init = 0.00001
    
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
    model = ResNet34_Conf_Localizer2()
    
    # load checkpoint (must transfer to CPU)
    # cp = "cpu_detrac_resnet34_alpha.pt"
    # cp = torch.load(cp)
    # model.load_state_dict(cp['model_state_dict'])

    if MULTI:
        model = nn.DataParallel(model,device_ids = [0,1])
    model = model.to(device)
    print("Loaded model.")
    
    
    with open(kf_params_file,"rb") as f:
        kf_params = pickle.load(f)
    
    # 3. create training params
    params = {'batch_size' : 32,
              'shuffle'    : True,
              'num_workers':0,
              'drop_last'  : True
              }
    
    # 4. create dataloaders
    try:   
        len(train_data)
        len(test_data)
    except:   
        train_data = Track_Dataset(train_image_dir, label_dir,n = 4)
        test_data =  Track_Dataset(test_image_dir,label_dir, n = 4)
        
    trainloader = data.DataLoader(train_data, **params)
    testloader = data.DataLoader(test_data, **params)
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    print("Got dataloaders. {},{}".format(datasizes['train'],datasizes['val']))
    
    # 5. define stochastic gradient descent optimizer    
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    #optimizer = optim.SGD(model.parameters(), lr = lr_init, momentum = 0.3)
    # 6. decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = True, mode = "min", patience = 1, factor=0.3)
    
    # 9. define losses
    losses = {"cls": [nn.CrossEntropyLoss()],
              "reg": [Box_Loss(), Conf_Loss()]
              }
    
    if True:    
    # train model
        print("Beginning training.")
        model = train(model,
                    optimizer, 
                    exp_lr_scheduler,
                    losses,
                    dataloaders,
                    device,
                    patience = patience,
                    kf_params = kf_params
                    )

   
