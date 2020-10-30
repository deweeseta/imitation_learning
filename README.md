# imitation_learning
Based on work by [naokishibuya](https://github.com/naokishibuya/car-behavioral-cloning) and [markmaz](https://markmaz.com/neet/imitation_learning/)

## Introduction

This lab provides an introduction to *end-to-end imitation learning for vision-only navigation* of a racetrack. Let's break that down:

- We will train a deep learning model - specifically, a convolutional neural network (CNN) - to regress a steering angle directly from an image taken from the "front bumper" of a car.
- Here, "imitation learning" refers to a branch of supervised machine learning which focuses on imitating behavior from human-provided examples. In our case, we will drive a car around a track several times to provide examples for the CNN to mimic. This learning objective is also frequently termed behavioral cloning.
  - We will contrast this with our next lab on "reinforcement learning" where a robot agent learns to accomplish a goal via exploration, not via examples.
- "Vision-only" refers to using an RGB camera as the only input to the machine learning algorithm.
  - LIDAR, depth, or vehicle IMU data are not used.
- Here, "end-to-end learning" is shorthand for the CNN's ability to regress a steering angle (i.e., an actuation for the Ackermann steering controller) from unprocessed input data (pixels). We will not need to pre-process input features ourselves, such as extracting corners, walls, floors, or optical flow data. The CNN will learn which features are important, and perform all the steps from image processing to control estimation itself ("end-to-end", loosely speaking).

We will drive a simulated car around a virtual racetrack and collecting camera data from the rendered game engine, as well as our game inputs. We will define a CNN that will regress similar game inputs in order for the car to complete the same track autonomously. 

The network is based on [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been proven to work in this problem domain.

|Lake Track|Jungle Track|
|:--------:|:------------:|
|[![Lake Track](images/lake_track.png)](https://youtu.be/hTPADovdyfA)|[![Jungle Track](images/jungle_track.png)](https://youtu.be/mZOc-zrbnR8)|
|[YouTube Link](https://youtu.be/hTPADovdyfA)|[YouTube Link](https://youtu.be/mZOc-zrbnR8)|


### Files included

- `train_pilotnet.ipynb` iPython Notebook used to create and train the model.
- `drive.py` The script to drive the car.
- `utils.py` The script to provide useful functionalities
- `model.h5` A pre-trained model weights.
- `environments.yml` conda environment (Use TensorFlow without GPU)

Note: drive.py is originally from [the Udacity Behavioral Cloning project GitHub](https://github.com/udacity/CarND-Behavioral-Cloning-P3) but it has been modified to control the throttle.
