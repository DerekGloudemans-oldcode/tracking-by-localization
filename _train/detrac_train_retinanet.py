"""
Derek Gloudemans - August 4, 2020
This file contains a simple script to train a retinanet object detector on the UA Detrac
detection dataset.
- Pytorch framework
- Resnet-50 Backbone
- Manual file separation of training and validation data
- Automatic periodic checkpointing
"""

### Imports

import os
import sys,inspect
import numpy as np
import random 
import math
import time
import cv2
random.seed = 0

import cv2
from PIL import Image
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms,models
from torchvision.transforms import functional as F
import matplotlib.pyplot  as plt
import collections


# add all packages and directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parent_dir)

from config.data_paths import directories
for item in directories:
    sys.path.insert(0,item)

from _data_utils.detrac.detrac_detection_dataset import Detection_Dataset, class_dict, collate
from config.data_paths import data_paths

# surpress XML warnings
import warnings
warnings.filterwarnings(action='once')

