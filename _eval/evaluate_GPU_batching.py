#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:45:21 2020

@author: worklab
"""

import matplotlib.pyplot as plt
import _pickle as pickle

with open("localize_1080.cpkl","rb") as f:
    results_1080 = pickle.load(f)

with open("localize_Quadro.cpkl",'rb') as f:
    results_quadro = pickle.load(f)


# plot throughput (at each batch size, how many objects per second can be tracked?)
throughput_1080 = [results_1080["batch_sizes"][i]*1000/results_1080["localize_times"][i] for i in range(len(results_1080["localize_times"]))]
throughput_quadro = [results_quadro["batch_sizes"][i]*1000/results_quadro["localize_times"][i] for i in range(len(results_quadro["localize_times"]))]    
batch_sizes = results_1080["batch_sizes"][:len(throughput_1080)]

plt.figure()
plt.plot(batch_sizes,throughput_1080)
plt.plot(batch_sizes,throughput_quadro)
plt.legend(["GTX_1080", "Quadro RTX 6000"]) 
plt.xlabel("Batch Size")
plt.ylabel("Throughput (224x224 images per second)")
plt.title("Throughput comparison")

# plot latency (at each batch size, how long does the batch take to process (batches per second))
latency_1080 = [1000/results_1080["localize_times"][i] for i in range(len(results_1080["localize_times"]))]
latency_quadro = [1000/results_quadro["localize_times"][i] for i in range(len(results_quadro["localize_times"]))]

plt.figure()
plt.plot(batch_sizes,latency_1080)
plt.plot(batch_sizes,latency_quadro)
plt.legend(["GTX_1080", "Quadro RTX 6000"]) 
plt.xlabel("Batch Size")
plt.ylabel("Batch Speed")
plt.title("Batch Latency Comparison")

# plot speedup
latency_speedup = [latency_quadro[i]/latency_1080[i] for i in range(len(latency_quadro))]
throughput_speedup = [throughput_quadro[i]/throughput_1080[i] for i in range(len(throughput_quadro))]
plt.figure()
plt.plot(batch_sizes,throughput_speedup)
plt.title("Speedup")