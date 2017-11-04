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

**Proportional Control**
###
When Kp is large, we reach out set point quickly, but overshoot it by more, and take longer to reach steady state (tends to oscillate over and under the target height for longer)

Conversely, when Kp is small, we take longer to reach the desired altitude, overshoot it by less, and it reaches steady state more quickly.

If Kp is too small, we may get the quadcopter off the ground, but we do not reach the desired altitude. Control effort is inefficient with high values of Kp. With larger Kp values, the steady offset and overshoot are also larger.

**Integral Control**
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/5984cc6e_pi-control-slide1/pi-control-slide1.png)