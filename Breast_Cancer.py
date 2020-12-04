#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:49:08 2020

@author: anthony
"""


import pandas as pd
import json
from PIL import Image
import numpy as np
'''
import matplotlib.pyplot as plt
plt.imshow([(255, 0, 0)])
plt.show()
'''

'''
Things to solve:
    - methodically access all images and json files
    - find a way to identify a tumor based on individual RBG pixels
        - try local averaging method, perhaps split image into small squares
        - try neural networks...
'''

# Reading the json as a dict
with open("BreCaHAD/groundTruth/Case_1-01.json") as json_data:
    data = json.load(json_data)

diagnoses = list(data.keys())
data_frame_dict = dict.fromkeys(diagnoses, [])

for i in range(len(diagnoses)):
    if len(data[diagnoses[i]]) > 0:
        x_list = []
        y_list = []
        for j in range(len(data[diagnoses[i]])):
            instance = data[diagnoses[i]][j]
            x_list.append(instance['x'])
            y_list.append(instance['y']) 
        data_frame_dict[diagnoses[i]] = pd.DataFrame({'x': x_list,'y': y_list})

#print(data_frame_dict['tumor'])

filename = "BreCaHAD/images/Case_1-01.tif"
# open image using PIL
img = Image.open(filename)

# convert to numpy array
img = np.array(img)

# find number of channels
if img.ndim == 2:
    channels = 1
    print("image has 1 channel")
else:
    channels = img.shape[-1]
    print("image has", channels, "channels")
    

'''
with open("BreCaHAD/images/Case_1-01.tif") as myfile:
    data = myfile.readlines()
'''