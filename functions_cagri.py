# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:41:55 2018

@author: cagri
"""
import numpy as np
def convolution(images,filter):
    """Gets an image and apply filters to them. With that, CNN tries to recognize patterns
    """
    convolution_output = np.zeros((images.shape[0],images.shape[1],len(filter)))
    image_padded = np.zeros((images.shape[0] + 2, images.shape[1] + 2))
    image_padded[1:-1, 1:-1] = images
    for z in range(len(filter)):
        for x in range(images.shape[0]):     # Loop over every pixel of the image
            for y in range(images.shape[1]):
            # element-wise multiplication of the filter and the image
                convolution_output[x,y,z]=(filter[z]*image_padded[x:x+3,y:y+3]).sum()      
    return convolution_output
   
def reLU(images):
    """Gets every individual pixel of given images and if the value of the pixel
        is negative we swap it to zero. With that, CNN stay mathematically healty and
        its not getting stuck near zero or goes infinity."""
    reLU_output = np.zeros((images.shape[0],images.shape[1],images.shape[2]))
    for k in range(images.shape[2]):
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                if images[i,j,k]<0:
                    reLU_output[i,j,k]=0
                reLU_output[i,j,k]= images[i,j,k]
    return reLU_output     

