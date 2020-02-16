# CNN_Classification_Ground_Up
Update: Neural Network Algorithm has been implemented in Neural Network Course and Keras part of the code has updated to our own dense and activation layer, forward and backpropogation, train and test functions. New version will be uploaded soon.
---
Update2: Apparantly we have learned a lot after this homework. Cagri Kilic and I worked for 6 months in the same company. We had developed several Computer Vision proof of concept projects using Matrox Imaging Library and Tensorflow API. We learned the core concepts of machine learning, deep learning and computer vision like we hoped before. We had taken Neural Networks, Statistical Data Processing, Data Mining courses along the way and now we are working on a autonomous car project using Deep Q Learning for our semester project. We have not much time to beautify the code and upload the codes for our projects. Several projects are on the way. Stay tuned.
---

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
  
Starting to code without any knowledge of deeplearning. We will learn along the way.

