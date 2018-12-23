# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:43:36 2018

@author: furkan
"""

import numpy as np
import skimage.io as io
from skimage import color as clr
from skimage.transform import resize
import matplotlib.pyplot as plt
import os

"""
# use for display
img = io.ImageCollection('./Dataset/cup/*.jpg')
for i in range(len(img)):
    plt.imshow(img[i])
    plt.title(img[i]+ ' '+ str(j))
    plt.show()
"""


root_folder = './Dataset'
class_names = next(os.walk(root_folder))[1]
image_dataset = np.empty( (2, len(class_names)), dtype=object)
image_dataset[0] = class_names
i = 0
for folder in class_names:
    j = 0
    image_dataset[1][i] = []
    for filename in os.listdir(root_folder + '/' +folder):
        path = root_folder + '/' + folder + '/' + filename
        image_dataset[1][i].append(io.imread(path, as_gray = True))
        image_dataset[1][i][j] = resize(image_dataset[1][i][j], (200, 200),anti_aliasing=True)
        j += 1
    i += 1
# test is a matrix of [10,n] that has %20 of every class
test = []

for i in range(len(image_dataset[1])):
    test.append(image_dataset[1][i][int(len(image_dataset[1][i])*0.8):])
    image_dataset[1][i] = image_dataset[1][i][:int(len(image_dataset[1][i])*0.8)]

# dataset[0]    dataset[1] %80
    
# airplanes     img_airplane[]
# butterfly     img_butterfly[]
# car_side      img_car_side[]
# cellphone     img_cellphone[]
# cup           img_cup[]
# dolphin       img_dolphin[]
# headphone     img_headphone[]
# laptop        img_laptop[]
# Motorbikes    img_Motorbikes[]
# pizza         img_pizza[]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    