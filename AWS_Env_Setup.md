
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