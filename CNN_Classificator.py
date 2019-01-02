import numpy as np
import skimage.io as io
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

## 
def one_hot_encode(labels):

# we use one hot encode method to represent the classes
    """
        input labels is a list of labels
        encoded is one hot encoding matrix (number of labels, number of class)
            ------------------------------------------
           |   1 0 0 0 0 0 0 0 0 0  -  0 airplanes   |
           |   0 1 0 0 0 0 0 0 0 0  -  1 butterfly   |
           |   0 0 1 0 0 0 0 0 0 0  -  2 car_side    |
           |   0 0 0 1 0 0 0 0 0 0  -  3 cellphone   |
           |   0 0 0 0 1 0 0 0 0 0  -  4 cup         |
           |   0 0 0 0 0 1 0 0 0 0  -  5 dolphin     |
           |   0 0 0 0 0 0 1 0 0 0  -  6 headphone   | 
           |   0 0 0 0 0 0 0 1 0 0  -  7 laptop      |
           |   0 0 0 0 0 0 0 0 1 0  -  8 motorbikes  |
           |   0 0 0 0 0 0 0 0 0 1  -  9 pizza       | 
            ------------------------------------------
    """
    encoded = np.zeros((len(labels), 10))
    for idx, val in enumerate(labels):
        encoded[idx][val] = 1
    return encoded

# root folder of the image database
root_folder = './Dataset' 

# name of the class folders as class_names
class_names = next(os.walk(root_folder))[1]  

image_dataset = np.empty( (2, len(class_names)), dtype=object)

image_dataset[0] = class_names
print("Files are being read")
i = 0
for folder in class_names:
    j = 0
    image_dataset[1][i] = []
    for filename in os.listdir(root_folder + '/' +folder):
        path = root_folder + '/' + folder + '/' + filename
        image_dataset[1][i].append(io.imread(path))
        
        # values of the images ranges between [0-1] 
        #so we dont need to do preprocessing on their values before feeding to tf
        
        image_dataset[1][i][j] = resize(image_dataset[1][i][j], (200, 200,3),anti_aliasing=True)
        j += 1
    random.shuffle(image_dataset[1][i])
    i += 1
    print("|", end = str(i))


"""
  ---------------------------------------
  |  dataset[0]  |  dataset[1]          |
  ---------------------------------------
  |  airplanes   |  img_airplane[]      |
  |  butterfly   |  img_butterfly[]     |
  |  car_side    |  img_car_side[]      |
  |  cellphone   |  img_cellphone[]     |
  |  cup         |  img_cup[]           |
  |  dolphin     |  img_dolphin[]       |
  |  headphone   |  img_headphone[]     |
  |  laptop      |  img_laptop[]        |
  |  Motorbikes  |  img_Motorbikes[]    |
  |  pizza       |  img_pizza[]         |
  ---------------------------------------

"""


# train for train images
# train_labels for the class names
train = []
train_labels = []

# test for test images
# test_labels for comparing the outputs to actual labels of them
test = []
test_labels = []

# seperating for training and testing with labels of them in a seperate array
# look trough every class and take the first 80 percent of images
# we take first 80 percent but we are sure that is is random because 
# we had shuffled every class in itself once we read them

for i in range(len(image_dataset[1])):
    for j in range(len(image_dataset[1][i])):
        if(j<int(len(image_dataset[1][i])*0.8)):
            train.append(image_dataset[1][i][j])
            train_labels.append(i)
        else:
            test.append(image_dataset[1][i][j])
            test_labels.append(i)

# we need to reshape our train and test arrays to make them numpy array and 
# put the indices of the images to the first column

train = np.array(train).reshape(-1,200,200,3) 
test  = np.array(test).reshape(-1,200,200,3)

# we need to make the labels numpy array as well

train_labels =np.array(train_labels)
test_labels =np.array(test_labels)

# we need indices array to shuffle them efficiently

indices_train = np.arange(train.shape[0])
indices_test  = np.arange(test.shape[0])

# to shuffle we shuffle the indices

np.random.shuffle(indices_train)
np.random.shuffle(indices_test)

# then we assign the indices to the elements of the array for train and test
# seperately

train = train[indices_train]
train_labels = train_labels[indices_train]

test = test[indices_test]
test_labels = test_labels[indices_test]

# we use one hot coding to represent our classes
encoded=one_hot_encode(train_labels)
encoded_test=one_hot_encode(test_labels)

    

