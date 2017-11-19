[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Project 4: Deep Learning  - Follow Me Quadcopter
##### Udacity Robotics Nanodegree
###### November 2017


![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59b9b854_following/following.png)


### Overview

***The goal of this project is to program a Quadcopter to identify a target and follow it.***

##### Specifically, the objectives of this project are to:

**1.** Analyze a stream of images coming from our front-facing camera on the drone

**2.** Classify each pixel using a Fully Convolutional Neural Network

**3.** Locate our target(the 'hero'/lady in red) in the pixels

**4.** Follow the target with the Quadcopter


We operate the quadcopter through **QuadSim** (ran locally).

The code for the segmentation network can be found at `model_training.ipynb`



This **README** is broken into the following sections: **Code Analysis, and Future Enhancements**.


### Code Analysis
The `model_training.ipynb` file is broken into individual sections, so that is how I will step through them.

##### Data Collection
Although I collected some of my own data, I opted to start with the images provided to us by Udacity and see what accuracy I could get. Again, those can be found [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip),  [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip), and [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip).

##### FCN Layers

###### Separable Convolutions
We were provided functions for separable convolutional layers, as well as convolutional layers for the `1x1`:
```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides, padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
    padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

###### Bilinear Upsampling

We're also provided an upsampling function that we'll use in the decoder layer:
```
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```
Here's a good dipiction of how upsampling works:

![alt text](https://www.researchgate.net/profile/Patrick_Van_der_Smagt2/publication/269577174/figure/fig2/AS:392180591022087@1470514547147/Fig-2-Upsampling-an-image-by-a-factor-of-2-Every-pixel-in-the-low-resolution-image-is.png)

Upsampling will be crucial in the second half of our FCN in order to transform the important features we learned from encoding into the final segmented image in the output.

##### Build the Model
Because we're challenged with the task of not just understanding **what** is in an image, but we need to figure out **where** the object is in the image, we're going to be using a network full of convolutional layers.

At first glance, we might think to just put many full-scaled convolutional layers one after another. This is good intuition, but we soon find this generates too many parameters, and becomes computationally very expensive. 

Instead, we'll use a method of first extracting important features through **encoding**, then upsample into an output image in **decoding**, finally assigning each pixel to one of the classes. 

Of course, every method has its disadvantages, and we'll lose some specific pixels during downsampling, and upsampling, but we'll also find out that this loss is negligible, and worth the increase in speed of computation.

###### Encoder Block
Harnessing the `separable_conv2d_batchnorm()` function, this was quite straight forward.
```
def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

###### 1x1 Convolution
Between the Encoder Block and the Decoder block, instead of having a 2D fully connected 1x1 layer--which is something we use in a regular ConvNet for classification--we opt to use a 1x1 convolutional layer. This is done so that we preserve spatial information. Since this is a task of segmentation, it is important to keep pixel data intact so we can upsample to generate an output image. 

This 1x1 convolution allows us to keep dimensionality low for the task of segmentation, without losing precious spatial data.

*Our new 1x1 convolution*
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59bb1431_1x1convolution/1x1convolution.png)

###### Decoder Block
For the decoder block, we harness `bilinear_upsample`, concatenation with `layers.concatenate`, and the `separable_conv2d_batchnorm` function.
```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    upsample = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([upsample, large_ip_layer])

    # Add separable convolution layers
    temp_layer = separable_conv2d_batchnorm(concat_layer, filters)
    output_layer = separable_conv2d_batchnorm(temp_layer, filters)
    
    return output_layer
```

###### Model

I decided to start with the symmetrical, 5 layer model architecture, like the one shown in the lecture. From what I've read, especially from recent research on networks like ***ResNet***, more layers are usually better, as long as they are not plain layers. But also, as networks become deeper, they take longer and longer to train. I figured 5 layers was a fair place to start. 

My network resembled something like this:
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59c7cad3_fcn/fcn.png)


For the first two layers, which comprise the **encoder**, and are in charge of determining the important features to extract from our input image, I chose `strides=2`, and the number of filters to be a power of 2: `2**6=64` for the first layer, then `2**7=128` for the second.
```
def fcn_model(inputs, num_classes):
    
    # ENCODER 
    layer1 = encoder_block(inputs, 64, 2)
    layer2 = encoder_block(layer1, 128, 2)
```
I went all the way to `2**8=256` for the middle **1x1** convolutional layer, so that it would be **1x1x256**, and we wouldn't compromise spatial data from our image.
```
    # 1x1 Convolution layer using conv2d_batchnorm()
    layer3 = conv2d_batchnorm(layer2, 256, kernel_size=1, strides=1)
```
The choice for the **decoder** was simple, in that I simply scaled back up from the middle **1x1** convolutional layer to the final output image size. In this moment, the term deconvolution feels fitting, as it is what I had in mind while building the decoder(however, I know it is a contentious term).
```
    # DECODER
    layer4 = decoder_block(layer3, layer1, 128)
    layer5 = decoder_block(layer4, inputs, 64)
```
We finally apply our favorite activation function, **Softmax**, to generate the scores for each of the classes:
```
    # OUTPUT
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(layer5)
```

##### Training

###### Hyperparameters
After trying a variety of values for `num_epochs` and having it set too high -- and therefore having training take multiple hours on AWS, I decided on the following training values:
```
learning_rate = 0.001
batch_size = 64
num_epochs = 50
steps_per_epoch = 65
validation_steps = 50
workers = 120
steps_per_epoch_check = 4132/batch_size
print(steps_per_epoch_check)
```
Most of these are quite standard, but some of the noteable ones are `steps_per_epoch_check`, `steps_per_epoch`, `workers`, and the `learning_rate`. 

`steps_per_epoch_check` is a verification parameter I created in order to set `steps_per_epoch`. In the Jupyter notebook provided to us, it mentioned a good heuristic for setting `steps_per_epoch` is by taking the number of training images, and dividing by the `batch_size`.

I determined `steps_per_epoch` by using `steps_per_epoch_check = 4132/batch_size`, where 4132 is the number of training images. This came out to 64.XX, so I opted for `steps_per_epoch = 65`.

For `workers`, I chose an arbitrarily high number, since we are, afterall, paying AWS to do the computations, I figured I should use alll of the processes they'll give me.

The `learning_rate` was reasonably straight forward to set. I initially started with `0.0005`, but found that the error was not decreasing fast enough, and it would take a very long time to train the network at this pace. So I opted for a rate which we have seen a good amount in the lecture of `0.001`.

After some tuning of parameters, starting long epochs, then stopping. My first `epoch` looked something like this:
![alt text](https://github.com/chriswernst/FollowMe-DL-Udacity-RoboticsND-Project4/blob/master/images/epoch1.png?raw=true)

After 10 `epochs`, the error was decreasing greatly, really flattening out:
![alt text](https://github.com/chriswernst/FollowMe-DL-Udacity-RoboticsND-Project4/blob/master/images/epoch10.png?raw=true)

With these parameters, anything beyond 50 epochs would be excessive and wasteful. Each epoch was running about 150-170s on average, which is much better than on my local MacBookPro of ~ 800 seconds per epoch.

##### Evaluation
For the first evaluation, with `learning_rate=0.001` I stopped the script after 10 epochs with 200 steps(because this would have taken all day, and cost me a lot of $$$). The loss was down to 0.03, and the `final_score` came out as **~0.395**. I was encouraged by this, so I continued training with more epochs.

Next, I adjusted the steps to 65, kept the `learning_rate=0.001` and the number of epochs to 50 -- because this should yield decent accuracy, but not take all day. The `final_score` came out as **~0.43**, Success! 

### Future Enhancements

First and foremost, adding data to train on would be a significant improvement.
After we've added more images, it would be wise to increase the number of epochs, and potentially decrease the learning rate -- assuming we have a lot of time/money/computational resources.

This addition of data and training time would allow the network to classify the target more accurately.

It also might be a smart idea to make the network deeper. From the research proven with ResNet /  GoogLeNet, adding more convolutions with skipped connections could prove to be a very good idea.

In order for this Deep Neural Network to be used to follow another target: such as a cat or a dog, it would just need to be trained on a new set of data. Also, the encoder and decoder layer dimensions may have to adjusted depending on the input pixel size.





