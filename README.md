#Self-driving car in a simulator with a tiny neural network

This is a solution for the Behavioral Cloning project of Udacity Self-Driving Car Nanodegree. This solution uses a tiny neural network with only 63 parameters.

Video of the actions of this neural network are here:
[Track 1](https://www.youtube.com/watch?v=AFHtBDaqQqk&t=10s)
[Track 2](https://www.youtube.com/watch?v=Emzy_Phz43g&t=14s)


##Model architecture

The model contains 6 layers:
1. Normalization layer
2. 2D convolution with kernel size of (3,3), valid padding and relu activation.
3. Max pooling layer with kernel size of (4,4) and valid padding.
4. Flatten layer.
6. Dense layer with 1 neuron to sum up the ouput data and produce the steering angle.

## Training

The model is trained with a batch size of 128 and epoch of 10.
