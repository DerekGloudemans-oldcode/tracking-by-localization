#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:30:54 2020

@author: worklab
"""


from detrac_resnet34_localizer import ResNet34_Localizer
import time
import torch



device = torch.device("cuda:0")

localizer = ResNet34_Localizer()
localizer = localizer.to(device)
localizer.eval()

transfer_times = []
localize_times = []
batch_sizes = [1,3,6,10,15,30,40,50,60,75,90,100,125,150,175,200,250,300,400,500,600,800,1000]

for b in batch_sizes:
    transfer_time = 0
    localize_time = 0
    for i in range(0,1000):
        data = torch.randn([b,3,224,224])
        
        start = time.time()
        data = data.to(device)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        transfer_time += elapsed
        
        with torch.no_grad():
            start = time.time()
            localizer(data)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            localize_time += elapsed
    
    print("Finished b = {}".format(b))
    transfer_times.append(transfer_time)
    localize_times.append(localize_time)
    