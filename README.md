[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Project 4: Deep Learning  - Follow Me Quadcopter
##### Udacity Robotics Nanodegree
###### November 2017

###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59b9b854_following/following.png)
###

### Overview

###### The goal of this project is to program a Quadcopter to identify a target and follow it.
###
###


##### Specifically, the objectives of this project are to:

**1.** Analyze a stream of images coming from our front-facing camera on the drone
###
**2.** Classify each pixel using a Fully Convolutional Neural Network
###
**3.** Locate our target(the 'hero'/lady in red) in the pixels
###
**4.** Follow the target with the Quadcopter
###
###


###

We operate the quadcopter through **QuadSim** (ran locally).

The code for the segmentation network can be found at `model_training.ipynb`

*Quick note on naming convention:* `THIS_IS_A_CONSTANT` *and* `thisIsAVariable`

This **README** is broken into the following sections: **Environment Setup, Code Analysis, Future Enhancements, and Background Information**.

###
###
###

### Environment Setup
###
This project will require `Python 3.5 64-bit` with anaconda, and `tensorflow 1.2.1`. Assuming the conda environment you deploy with Python 3.5 is called `RoboND`, install the required dependencies with:
```
$ source activate RoboND
$ pip install tensorflow==1.2.1
$ pip install socketIO-client
$ pip install transforms3d
$ pip install PyQt5
$ pip install pyqtgraph
```
###
This will install the CPU version of `tensorflow` only.
###

Clone the project repository:
```
$ git clone https://github.com/udacity/RoboND-DeepLearning-Project.git
```

###

Download the simulator [here](https://github.com/udacity/RoboND-DeepLearning-Project/releases/tag/v1.2.2)

###
Download the Data:

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip)
[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)
[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

###


#### AWS Setup
For training, we'll harness the GPU computing resources available through AWS. We'll need `1` instance of the `px2.large`

A good walkthrough of how to ssh into the AWS GPU and run a Jupyter notebook is [here](https://towardsdatascience.com/setting-up-and-using-jupyter-notebooks-on-aws-61a9648db6c5) and the video of it is [here.](https://www.youtube.com/watch?time_continue=160&v=q1vVedHbkAY)

##### Steps to ssh into AWS Linux Machine:
First, change your directory to where `.pem` file is. Then, in a terminal:
```
ssh -i "yourAWSkey.pem" ubuntu@ec2-54-202-123-251.us-west-2.compute.amazonaws.com
```
Where the `ec2-XX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com` is the Public DNS (IPv4) can be found on the AWS EC2 management page. *Note, this DNS address changes each time you start an instance!*

Now, Launch a jupyter notebook:
```
jupyter notebook --ip='*' --port=8888 --no-browser
```

Open a new terminal that is pointing locally, and type:
```
ssh -i "yourAWSkey.pem" -L 8212:localhost:8888 ubuntu@ec2-54-202-123-251.us-west-2.compute.amazonaws.com
```

Next, we should be able to navigate locally on the browser, and type:
```
localhost:8888
```

Finally, launch the the `model_training.ipynb`

###
### Code Analysis
The `model_training.ipynb` file is broken into individual sections, so that is how I will step through them.
###
##### Data Collection
Although I collected some of my own data, I opted to start with the images provided to us by Udacity and see what accuracy I could get. Again, those can be found [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip),  [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip), and [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip).
###
##### FCN Layers
###
###### Separable Convolutions
We were provided functions for separable convolutional layers, as well as fully connected convolutional layers for the `1x1`:
```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,         padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
    padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```
###
###### Bilinear Upsampling
###
We're also provided an upsampling function that we'll use in the decoder layer:
```
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```
Here's a good dipiction of how upsampling works:
###
![alt text](https://www.researchgate.net/profile/Patrick_Van_der_Smagt2/publication/269577174/figure/fig2/AS:392180591022087@1470514547147/Fig-2-Upsampling-an-image-by-a-factor-of-2-Every-pixel-in-the-low-resolution-image-is.png)
###
##### Build the Model
###
###### Encoder Block
Harnessing the `separable_conv2d_batchnorm()` function, this was quite straight forward.
```
def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

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
###
###### Model
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59c7cad3_fcn/fcn.png)
Since our incoming image is `256x256` pixels (which is nicely divisible by 8), it makes sense to build our network as a 5 layer model
###
Encoder:
![alt text](https://github.com/chriswernst/FollowMe-DL-Udacity-RoboticsND-Project4/blob/master/images/Encoder.JPG?raw=true)
###
Decoder:
![alt text](https://github.com/chriswernst/FollowMe-DL-Udacity-RoboticsND-Project4/blob/master/images/Decoder.JPG?raw=true)
###
###
Model:
```
def fcn_model(inputs, num_classes):
    
    # ENCODER 
    layer1 = encoder_block(inputs, 64, 2)
    layer2 = encoder_block(layer1, 128, 2)

    # Fully connected 1x1 Convolution layer using conv2d_batchnorm().
    layer3 = conv2d_batchnorm(layer2, 256, kernel_size=1, strides=1)
    
    # DECODER
    layer4 = decoder_block(layer3, layer1, 128)
    layer5 = decoder_block(layer4, inputs, 64)
    
    # OUTPUT
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(layer5)
```

##### Training
###
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
Most of these are quite standard, but some of the noteable ones are `steps_per_epoch`, and `workers`. 

I determined `steps_per_epoch` by using `steps_per_epoch_check = 4132/batch_size`, where 4132 is the number of training images. This came out to 64.XX, so I opted for `steps_per_epoch = 65`.

For `workers`, I chose an arbitrarily high number, since we are, afterall, paying AWS to do the computations, I figured I should use alll of the processes they'll give me.

After some tuning of parameters, starting long epochs, then stopping. My first `epoch` looked something like this:
![alt text](https://github.com/chriswernst/FollowMe-DL-Udacity-RoboticsND-Project4/blob/master/images/epoch1.png?raw=true)

After 10 `epochs`, the error was decreasing greatly, really flattening out:
![alt text](https://github.com/chriswernst/FollowMe-DL-Udacity-RoboticsND-Project4/blob/master/images/epoch10.png?raw=true)

With these parameters, anything beyond 50 epochs would be excessive and wasteful. Each epoch was running about 150-170s on average, which is much better than on my local MacBookPro of ~ 800s per epoch.
###
##### Evaluation
For the first evaluation, I stopped the script after 10 epochs with 200 steps(because this would have taken all day, and cost me a lot of $$$). The loss was down to 0.03, and the `final_score` came out as **~0.395**. I was encouraged by this, so I continued training with more epochs.

Next, I adjusted the steps to 65, and the number of epochs to 50 -- because this should yield decent accuracy, but not take all day. The `final_score` came out as **~0.43**, Success! 


###
### Future Enhancements
###
First and foremost, adding data to train on would be a significant improvement.
After we've added more images, it would be wise to increase the number of epochs, and potentially decrease the learning rate -- assuming we have a lot of time/money/computational resources.

This addition of data and training time would allow the network to classify the target more accurately.

In order for this Deep Neural Network to be used to follow another target: such as a cat or a dog, it would just need to be trained on a new set of data. Also, the encoder and decoder layer dimensions may have to adjusted depending on the input pixel size.
###
###
###
### Course Background Information
###
##### PID Controller
Proportional, Integral, and Derivative control is what we'll be using to control our quad copter.

**Proportional Control**
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/5956d8b9_eq2/eq2.png)
###
When Kp is large, we reach out set point quickly, but overshoot it by more, and take longer to reach steady state (tends to oscillate over and under the target height for longer)

Conversely, when Kp is small, we take longer to reach the desired altitude, overshoot it by less, and it reaches steady state more quickly.

If Kp is too small, we may get the quadcopter off the ground, but we do not reach the desired altitude. Control effort is inefficient with high values of Kp. With larger Kp values, the steady offset and overshoot are also larger.

**Integral Control**
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/5984cc6e_pi-control-slide1/pi-control-slide1.png)
###

When Ki is too large, we arrive at our set point quickly. Although, there is too much oscillation and we overshoot our set point by a larger margin. It also takes longer to arrive at a steady state. Steady state offset at 30seconds remains high.

When Ki is too small, we may not even arrive fully at our set point. The oscillations are more calm, and steady state offset is smaller.



###
**Derivative Control**
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/5959aca0_eq6/eq6.png)
###
The derivative term attempts to “predict” what the error will be by linearly extrapolating the change in error value. Remember, the derivative is the slope of the line tangent to the curve.

When Kd is too large, rise time is short, overshoot is high, and control effort is inefficient. 

When Kd is too small, rise time is larger, overshoot still is quite high, and control effort is inefficient. 

At the appropriate Kd value, oscillations are dampened, leaving the control effort efficient, rise time short, and overshoot lower. 


**PID** 
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/598dce8b_building-a-pid-controller-01/building-a-pid-controller-01.png)
This PID Controller is much better than PD control because it actually reaches our set point. The three parameters help to reach the set point, aggregate the error, and then smooth the curve. The control effort also looks very efficient!
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/598a2eae_09-pid-control-summary-e05/09-pid-control-summary-e05.png)


#### Gradient Descent
We can attempt to optimize the Kp, Ki, and Kd parameters in a variety of ways. One popular method is using Gradient Descent!
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/5991ebb9_13-l-tuning-strategies-01/13-l-tuning-strategies-01.png)
###

##### Cascade PID Controller
Cascaded controllers can help us execute multiple processes simultaneous. It is typical practice for the inner loop to be running at 10x the speed of the outer loop.
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/5991eaea_07-cascade-pid-control/07-cascade-pid-control.png)
###
We'll now be builing our own Cascade PID Controller!
###

#### Controls Lab

We'll be building from the [Controls Lab Repo!](https://github.com/udacity/RoboND-Controls-Lab)

Install the Repo locally(I've done so to the Linux VM). Next, download the latest version of the Simulator for your host system ("DroneSim_OSX").

Alter the file `DroneSim_OSX.app/Contents/ros_settings.txt` (to do this on OSX, right-click the package and select "Show Package Contents") so that the IP address of the VM is correct and points to the Linux VM `"vm-ip" : "192.168.30.111"` and set `"vm-override" : true`.

Now, launch the application as normal from OSX.

###

#### Deep Learning

###
Neural Networks, or multilayer Perceptron(MLP), are at the core of Deep Learning. A Deep Neural Network, is what makes the learning *Deep*. This means the neural networks have more than 1 hidden layer, like this: 
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/March/58db494a_karpathy-network/karpathy-network.png)
###

Here is a really cool application of Deep Learning called style transfer!
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/November/59f92ba0_cat/cat.png)


We'll be leveraging the **Sigmoid, or Logistic Function**, to generate a continuous prediction, where `0 < yHat < 1`. Remember, the **Sigmoid** function is `1/(1+e^-x)`
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/58800a83_sigmoid/sigmoid.png)
###

A very simple neural network looks like this:
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/589366f0_simple-neuron/simple-neuron.png)
###

Sigmoid is great for when we need to determine a Yes/No classification problem--1 output layers. But When we have 3 or more classes, it is best to use the **Softmax** function. 
###

**SoftMax Function**
###
The softmax function is based on the exponential function. In the numerator we have `e**score`; and in the denominator, we have the summation of `e**score` for each of the observations: `e**score(1)` + `e**score(2)` + `...` + `e**score(n)`

###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58938e9e_softmax-math/softmax-math.png)
###

###
**Logistic Regression**
`logistic_regression_explanation` is how we derived the following graph:
###
![alt text](https://github.com/chriswernst/FollowMe-DL-Udacity-RoboticsND-Project4/blob/master/images/gradientDescentLearning.png?raw=true)
###
The data is made up and comes from `perceptron.csv`


###
##### TensorFlow

We'll be harnessing the Google Developed, **TensorFlow** to execute our Deep Learning algorithms:
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58116cd8_maxresdefault/maxresdefault.jpg)

###
First, we'll test it on the English alphabet a-j, in the dataset **notMNIST**:
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58051e40_notmnist/notmnist.png)

###

##### Convolutional Neural Networks
###


![alt text](https://cdn-images-1.medium.com/max/1200/1*1okwhewf5KCtIPaFib4XaA.gif)

###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377d67_vlcsnap-2016-11-24-15h52m47s438/vlcsnap-2016-11-24-15h52m47s438.png)

###


![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581a58be_convolution-schematic/convolution-schematic.gif)


###

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/5840fe04_retriever-patch-shifted/retriever-patch-shifted.png)

###
We can add zero padding to our image so we don't lose dimensionality
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5837d4ee_screen-shot-2016-11-24-at-10.05.46-pm/screen-shot-2016-11-24-at-10.05.46-pm.png)

###

Here is an example of what the CNN sees in each layer:

**Layer 1**
The first layer can detect diagonal lines(+45,-45)
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583cbd42_layer-1-grid/layer-1-grid.png)
### 
Even though they have different colors and textures
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583cbace_grid-layer-1/grid-layer-1.png)

###
**Layer 2**
Layer 2 is able to detect more complex shapes such as circles and stripes
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583780f3_screen-shot-2016-11-24-at-12.09.02-pm/screen-shot-2016-11-24-at-12.09.02-pm.png)

###
**Layer 3**
The third layer is able to pick out even more complex combinations of features from layer 2:
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5837811f_screen-shot-2016-11-24-at-12.09.24-pm/screen-shot-2016-11-24-at-12.09.24-pm.png)

###
**Layer 4/5**
The fifth and final layer picks out and groups these complex shapes together to form the highest order items we care about:
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58378151_screen-shot-2016-11-24-at-12.08.11-pm/screen-shot-2016-11-24-at-12.08.11-pm.png)

###
###### CNN Layer Architecture

###
LENET-5, Yan Lecun - 1998
###
![alt text](https://www.researchgate.net/profile/Haohan_Wang/publication/282997080/figure/fig10/AS:305939199610894@1449952997905/Figure-10-Architecture-of-LeNet-5-one-of-the-first-initial-architectures-of-CNN.png)


###
ALEXNET -  Alex Krizhevsky - 2012
###
![alt text](https://kratzert.github.io/images/finetune_alexnet/alexnet.png)


###### Suggestions for CNNs
From the Udacity Lecture: "We suggest you start off with a simple implementation by using a single convolutional layer with max-pooling, and a single fully-connected layer. Observe the loss and validation accuracy values you obtain. Then slowly refine your model by adding more layers, dropouts for regularization, tuning your hyperparameters etc. to achieve a good, high level of accuracy on your validation and test sets."
###


##### Fully Convolutional Neural Networks

FCNs have 2 parts -  encoder, and decoder:
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59c7cad3_fcn/fcn.png)

###

Dialated Convolutions(aka Atrous Convolutions):

![alt text](https://cdn-images-1.medium.com/max/1200/1*SVkgHoFoiMZkjy54zM_SUw.gif)
*2D convolution using a 3 kernel with a dilation rate of 2 and no padding*

###
Transposed Convolutions(Or Deconvolutions):
![alt text](https://cdn-images-1.medium.com/max/1200/1*BMngs93_rm2_BpJFH2mS0Q.gif)
*2D convolution with no padding, stride of 2 and kernel of 3*


###
![alt text](https://cdn-images-1.medium.com/max/1200/1*Lpn4nag_KRMfGkx1k6bV-g.gif)
*Transposed 2D convolution with no padding, stride of 2 and kernel of 3*








