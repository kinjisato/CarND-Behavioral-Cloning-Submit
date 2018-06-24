# **Behavioral Cloning** 

## Writeup

### Kinji Sato 24/June/2018

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/nVidia_model.png "nVidia model"
[image2]: ./images/mseLossVsEpochs.PNG "mse loss"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (same as original one)
* model.h5 containing a trained convolution neural network 
* writeup.md for summarizing the results (this document)
* video.mp4 video that car is running autonomous mode at my environment

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

![alt text][image1]

I used nVidia model for my neural network model. My neural network model structure is below.
I slotted some dropout layer into the model for the case of overfitting.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 66 x 200 x3 YUV image (normalized)  							| 
| Convolution 5x5     	| 2x2 stride, padding 'VALID', 24 depth 	|
| Activation					|	ReLU									|
| Convolution 5x5     	| 2x2 stride, padding 'VALID', 36 depth 	|
| Activation					|	ReLU									|
| Convolution 5x5     	| 2x2 stride, padding 'VALID', 48 depth 	|
| Convolution 3x3	    | 64 depth 	|
| Activation					|	ReLU									|
| Convolution 3x3	    | 64 depth 	|
| Activation					|	ReLU									|
| Flatten		|       									|
| (Dropout)					|	(0.5)									|
| Fully connected		| output = 100       									|
| (Dropout)					|	(0.5)									|
| Fully connected		| output = 50       									|
| (Dropout)					|	(0.5)									|
| Fully connected		| output = 10      									|
| (Dropout)					|	(0.5)									|
| Fully connected		| output = 1      									|


#### 2. Attempts to reduce overfitting in the model

As mentioned above, I slotted some dropout layers into nVidia model, those are prepared between fully connected layers after flatten. At the case of overfitting, those could be activated and tuned.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

This was big problem for me. Because, I'm not sure how much data volume required for training the model. As first, I used the data I could download from Udacity web. There were more than 8,000 lines in csv file, and each have 3 images (center, left and right), and I added flipped imaged (same as lecture video), so I could get 6 images from each lines. But unfortunately, the car behavior was not good. So I took the data from simulator by myself. Just only a few laps, and volume was the half of the data from Udacity. But, the car behavior was better.
So, I'm very confused. At least, the data I collected gave good result. But this is very difficult to explain the reason.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My approach was the same as the lecture video. And finally, I decided to use nVidia model, because that gave me better result than other filters. 
I checked the mean square error of training loss and validation loss. those were reduced by each epochs, and both loss values were almost the same. So I thought there was not overfitting with using this model and my data.

![alt text][image2]

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
