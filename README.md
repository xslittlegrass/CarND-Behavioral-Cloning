#Self-driving car in a simulator with a tiny neural network

This is a solution for the Behavioral Cloning project of Udacity Self-Driving Car Nanodegree. This solution uses a tiny neural network with only 63 parameters.

Video of the actions of this neural network are here:
[Track 1](https://www.youtube.com/watch?v=AFHtBDaqQqk)
[Track 2](https://www.youtube.com/watch?v=Emzy_Phz43g)

A post about this solution is at [Self-driving car in a simulator with a tiny neural network](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234#.outp5gx7f).

##Model architecture

The model contains 6 layers:
 1. Normalization layer
 2. 2D convolution with kernel size of (3,3), valid padding and relu activation.
 3. Max pooling layer with kernel size of (4,4) and valid padding.
 4. Flatten layer.
 6. Dense layer with 1 neuron to sum up the ouput data and produce the steering angle.

## Training

The model is trained with a batch size of 128 and epoch of 10 and an adam optimization method. Since the input images are resized to a dimension of 16X32, all the data can be fit into the memeory, and thus a generator is not need. Because of the small size of the network and input data, the model can be trained with just a few seconds. The reason to use only 10 epochs is that this tiny model converges very fast in just a few epoch, and the validation accuracy usually flattens out around 10 epochs. Using more than 10 epochs will not increase the validation accuracy. Testing is performed in the simulator. 

