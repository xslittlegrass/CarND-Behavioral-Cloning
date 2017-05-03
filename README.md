# Self-driving car in a simulator with a tiny neural network

This is a project for Udacity Self-Driving Car Nanodegree program. The aim of this project is to control a car in a simulator using neural network. This implementation uses a convolutional neural network (CNN) with only 63 parameters, yet performs well on both the training and test tracks. The implementation of the project is in the files `drive.py` and `model.py` and the explanation of the implementation is in `project-3.ipynb`. Videos of the actions of this neural network are here: [Track 1](https://www.youtube.com/watch?v=AFHtBDaqQqk) [Track 2](https://www.youtube.com/watch?v=Emzy_Phz43g). A post about this solution is at [Self-driving car in a simulator with a tiny neural network](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234#.outp5gx7f).

## How to run the project

This project needs the Udacity self-driving car simulator to run. It can be downloaded from [here](https://github.com/udacity/self-driving-car-sim). After setting up the simulator, choose the autonomous mode, and run

```
python drive.py model.json
```
in the terminal. The connection between the simulator and the controlling neural network will be taken care automatically.

## Project architecture

The project architecture is show in the following figure. The simulator capture the image through its onboard camera and send them to CNN. The CNN takes the image and send back the steering angle to the simulator.

<img src=./images/architecture.png width=500>

The car in the simulator can capture three images at a time, corresponding to the left, center and right cameras.

<img src=./images/camera_images.png width=500>

## CNN architecture

The CNN used in this project has only 6 layers with only 63 parameters:

 1. Normalization layer
 2. 2D convolution with kernel size of (3,3), valid padding and relu activation.
 3. Max pooling layer with kernel size of (4,4) and valid padding.
 4. Flatten layer.
 6. Dense layer with 1 neuron to sum up the output data and produce the steering angle.

Before entering the CNN, the image is resized to 16x32 and only the S channel of the image will be used.

<img src=./images/CNN_model.png width=500>

## Training

The training data comes from the labeled image recorded by the simulator, while driving the car around the training track by a human. Approximately 7000 images are used in training the model. In order to the teach the car to recover from the road, the labeled angles for the left and right images are shifted by a constant before used in training.

<img src=./images/shift_angle.png width=400>

The model is trained with a batch size of 128 and epoch of 10 and an adam optimization method. Since the input images are resized to a dimension of 16X32, all the data can be fit into the memory, and thus a generator is not need. Because of the small size of the network and input data, the model can be trained with just a few seconds. The reason to use only 10 epochs is that this tiny model converges very fast in just a few epoch, and the validation accuracy usually flattens out around 10 epochs. Using more than 10 epochs will not increase the validation accuracy.


