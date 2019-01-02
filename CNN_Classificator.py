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

#%%    making and training the model

# we took training data as batches 100 images 
batch_size = 100
num_classes = len(image_dataset[1])
# we chose 100 epochs because we dont have many images
epochs = 100
data_augmentation = True
num_predictions = 20


model = Sequential()

# convolutional layer applies different filters 
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(200,200,3)))

# reLu makes the negative values 0 and positive values stays the same
model.add(Activation('relu'))

# pooling makes the image smaller as it tooks the max value in 2 by 2 matrix
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# flatten takes all the outputs from all layers
# and converts it to [1,n] array
# dense layer is the neural network itself
# it does assign the weights and biases and with backpropagation it updates
# all the weigths, biases and filters
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])




datagen =   ImageDataGenerator(
                rotation_range=10.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.,
                zoom_range=1.,
                horizontal_flip=True,
                vertical_flip=True)

# we store loss and acc data to be able to plot the data 
history = model.fit_generator(datagen.flow(train, encoded, batch_size = batch_size),
                              steps_per_epoch  = int(np.ceil(train.shape[0] / float(batch_size))),
                              epochs = epochs,
                              validation_data = (test, encoded_test),
                              workers = 4)
 
model.evaluate(test, encoded_test)

# we can plot the history of losses and accuracy as we stored it
# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

#%%   for saving the model

# we show the loss and acc before we save it 
scores = model.evaluate(test, encoded_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# we save the model in json file and weights in hdf5 file

# serialize model to JSON
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./model.h5")
print("Saved model to disk")

#%%   for loading the model
# we can use this module to load the model and weights
# because we cant train it every single time 
# so once you train it you can still use the model to make predictions 

# load json and create model
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(test, encoded_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[0], score[0]*100))
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#%%
# after the test data we use our own data that we found online to calculate
# the accuracy ith foreign data 

test_self =[]
for filename in os.listdir('./test_self_images'):
        path = './test_self_images' + '/' + filename
        test_self.append(resize(io.imread(path), (200, 200,3),anti_aliasing=True))
test_self = np.array(test_self).reshape(-1,200,200,3)
total_answ = 0
for i in range(len(test_self)):
    plt.imshow(test_self[i])
    plt.title(image_dataset[0][loaded_model.predict_classes(test_self[i][np.newaxis,:])])
    plt.show()
    answ = int(input("1 for true, 0 for wrong -- "))
# we count the number of true images and compare it with 
# the total number of images to calculate the percentage of accuracy
    total_answ += answ
print("accurracy is " + str(float(total_answ/len(test_self))*100)+"%" )

    

