# CNN_Classification_Ground_Up

Image Classification using Convolutional Neural Network
-------------------------------------------------------
Using Tensorflow and Keras 
Image Database : https://drive.google.com/open?id=1bKSFyUVwREriUXKJegiCSpp7Qif-x1v   
   * Read Images and Labels                                                        |
   * Split the data to train and test                                              |
   * Make the model and train                                                      |
   * Save the model to use later                                                   |
   * Calculate the accuracy and loss with test images                              |
   * Find random images according to the classes and calculate accuracy and loss   |
   
---------------------------------------------------------------------------------


Update 1 : We found out that we have to use Tensorflow and Keras libraries to make the project but we are thrilled to write our functions            anyway. In the semester break we will try to update it with our functions and compare with Tensorflow and Keras.

  * Reading images in grayscale 
   update: agreed on RGB ++
   
  * Split data to train and test
   update: after shuffle inside the folders, we take first 80 percent of each folder to make our train data and take the last 20 percent              to make the test data then we shuffle it again because the train and test arrays are in the order which files are++
  
  * Construct the layers
   update: cagri built the convolutional and activation layer, furkan built the pooling layer and took the data and shaped into what we              need
   
  * Build the train function
   update: we still need the backpropagation
  
  * Build the loss function
   update: we still need the loss function
  * Do backpropogation according to loss function using gradient descent
   
  * Use test data to approximate the success
  
Starting to code without any knowledge of deeplearning we will learn along the way

