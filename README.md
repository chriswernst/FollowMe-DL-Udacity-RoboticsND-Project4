[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Project 4: Deep Learning  - Follow Me Quadcopter
##### Udacity Robotics Nanodegree
###### November 2017

###
###

### Overview

###### The goal of this project is to program a Quadcopter to identify a target and follow it.
###
###
***(Click the image for a video of the final result)***
[!][Click for the final project video](UPDATE with URL)](https://youtu.be/v3lSNYWOniU)

##### Specifically, the objectives of this project are to:

**1.** 
###
**2.** 
###
**3.** 
###
**4.** 
###
**5.** 
###
**6.** 
###
###


###

We operate the quadcopter through **ROS Kinetic** (ran through Linux Ubuntu 16.0) and commands are written in **Python**.

The code driving this project and interacting with ROS can be found at `TBD.py`

*Quick note on naming convention:* `THIS_IS_A_CONSTANT` *and* `thisIsAVariable`

This **README** is broken into the following sections: **Environment Setup, Code Analysis, and Debugging**.

###
###
###

### Environment Setup
###
*To be filled in*
###
### Code Analysis

#### PID Controller
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
Neural Networks are at the core of Deep Learning. We'll be leveraging the **Sigmoid, or Logistic Function**, to generate a continuous prediction, where `0 < yHat < 1`. Remember, the **Sigmoid** function is `1/(1+e^-x)`
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



##### TensorFlow

We'll be harnessing the Google Developed, **TensorFlow** to execute our Deep Learning algorithms:
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58116cd8_maxresdefault/maxresdefault.jpg)

###
First, we'll test it on the English alphabet a-j, in the dataset **notMNIST**:
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58051e40_notmnist/notmnist.png)



