#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:29:07 2020

@author: worklab
"""


# # 4 x Quadro RTX 6000 machine
# data_paths = {
#     "train_im":"/home/worklab/Data/cv/Detrac/DETRAC-train-data",
#     "train_lab":"/home/worklab/Data/cv/Detrac/DETRAC-Train-Annotations-XML-v3",
#     "test_im":"/home/worklab/Data/cv/Detrac/DETRAC-test-data",
#     "test_lab":"/home/worklab/Data/cv/Detrac/DETRAC-Test-Annotations-XML-v3",
#     "train_partition":"/home/worklab/Data/cv/Detrac/detrac_train_partition",
#     "val_partition":"/home/worklab/Data/cv/Detrac/detrac_val_partition"
#     }

# directories = ["/home/worklab/Documents/derek/tracking-by-localization/config",
#                "/home/worklab/Documents/derek/tracking-by-localization/data/detrac_detections",
#                "/home/worklab/Documents/derek/tracking-by-localization/_data_utils",
#                "/home/worklab/Documents/derek/tracking-by-localization/_data_utils/detrac",
#                "/home/worklab/Documents/derek/tracking-by-localization/_detectors",
#                "/home/worklab/Documents/derek/tracking-by-localization/_eval",
#                "/home/worklab/Documents/derek/tracking-by-localization/_localizers",
#                "/home/worklab/Documents/derek/tracking-by-localization/_train",
#                "/home/worklab/Documents/derek/tracking-by-localization/_tracker"             
#                ]




# 2 x GTX 1080 Ti machine
data_paths = {
    "train_im":"/home/worklab/Desktop/detrac/DETRAC-all-data",
    "train_lab":"/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3",
    "test_im":"/home/worklab/Desktop/detrac/DETRAC-test-data",
    "test_lab":"/home/worklab/Desktop/detrac/DETRAC-Test-Annotations-XML",
    "train_partition":"/home/worklab/Desktop/detrac/DETRAC-train-data",
    "val_partition":"/home/worklab/Desktop/detrac/DETRAC-val-data",
    "fast_partition":"/home/worklab/Desktop/detrac/DETRAC-short-data",
    "filter_params":"/home/worklab/Documents/code/tracking-by-localization/config/filter_params",
    "tracking_output":"/home/worklab/Documents/code/tracking-by-localization/data/tracking_outputs",
    "detections":"/home/worklab/Documents/code/tracking-by-localization/data/detrac_detections"
    }

directories = ["/home/worklab/Documents/code/tracking-by-localization/config",
               "/home/worklab/Documents/code/tracking-by-localization/data/detrac_detections",
               "/home/worklab/Documents/code/tracking-by-localization/_data_utils",
               "/home/worklab/Documents/code/tracking-by-localization/_data_utils/detrac",
               "/home/worklab/Documents/code/tracking-by-localization/_detectors",
               "/home/worklab/Documents/code/tracking-by-localization/_eval",
               "/home/worklab/Documents/code/tracking-by-localization/_localizers",
               "/home/worklab/Documents/code/tracking-by-localization/_train",
               "/home/worklab/Documents/code/tracking-by-localization/_tracker",
               "/home/worklab/Documents/code/tracking-by-localization/_detectors/pytorch_retinanet",
               "/home/worklab/Documents/code/tracking-by-localization/_eval/py_motmetrics",
               "/home/worklab/Documents/code/tracking-by-localization/_localizers/pytorch_retinanet_localizer"
               ]
