# **Behavioral Cloning** 

## Writeup, Juha-Matti Tirilä 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_architecture]: ./images/cnn_architecture.png "Model Visualization"
[flipped_image]: ./images/flipped_image.jpg "Flipped image"
[correcting_from_gray_edge]: ./images/correction_from_gray_edge.jpg "Recovery Image from different texture"
[correction_from_right]: ./images/correction_from_right.jpg "Recovery Image from right"
[correction_from_left]: ./images/correction_from_left.jpg "Recovery Image from left"
[center_lane_driving]: ./images/center_lane_driving.jpg "Center lane driving"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted and Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by 
executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the 
pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a slight modification of the Nvidia architecture presented in the lecture notes. It is a 
convolution neural network with first 5x5 and subsequently also 3x3 filter sizes and depths 
between 24 and 64 (model.py lines 54-58). These convolutional layers are followed by fully connected layers, 
starting with one containing 100 neurons, and in the subsequent layers the neuron count is gradually reduced to 1 
to get scalar output. 

The model includes RELU layers to introduce nonlinearity (lines 54-58), and the 
data is normalized in the model using a Keras lambda layer (code line 53). The normalization layer just zero-centers 
 the data and scales it between -0.5 and 0.5.
 
#### 2. Reducing overfitting in the model

The model contains four dropout layers in order to reduce overfitting (model.py lines 62 and 64). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 
70, validation split). The model was tested by running it through the simulator and ensuring that the vehicle 
could stay on the track. I modified the drive.py script used to operate the model file in the simulator a bit, 
increasing the target speed to 16 mph. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). I experimented with 
the number or epochs and soon discovered that an excessive number of epochs would result in overfitting according
to the validation set metrics, and also the driving behavior of the resulting model would be poor. 

Some further experimentation was performed on the number dropout probability of the dropout layers. In these 
experiments, it seemed as though having multiple dropout layers toward the output layer of the network was beneficial. 
Also, experiments with smaller dropout rate were unsuccessful so in the final submission, all the three dropout layers
have a dropout rate of 50%. 

Some further experimentation was used with the ImageDataGenerator of Keras. I tried out some rotation, transalation 
and whitening transformations. Another approach I tried was to use TensorFlow's contrast enhancement 
(tf.image.adjust_contrast). However, these latter experiments added complexity and the preliminary driving results 
were not better than without these normalizations / augmentations. Hence, I decided to not use these techiques in my 
submission. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and 
recovering from both sides of the road. The recovery actions were carefully recorded in order to just capture the 
corrective steering, not the erraneous actions before that.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

##### AlexNet

The overall strategy for deriving a model architecture was to first try experimenting with some of the advanced 
architectures like Alexnet.  

However, I soon run into problems with system resources also on AWS. Meanwhile, some participants reported they had 
trained models successfully using quite simple architectures so 
I also decided to let go of the idea of advanced models. 
  
##### Back to Basics  

I then decided to start with something simple, try different approaches and was hoping to finally settle on something 
that drives the track successfully. 
  
As a next step, kind of to achieve a reference level, I designed a simple multilayer perceptron model. The basic idea
was the same as in the lecture material: to get the generic pipeline for training and testing to work, without worrying
too much about the actual model. As expected, I got everything working so I was able to get the model driving, but 
the performance of the model was pretty bad. 

My next attempt was to experiment with different modifications of the LeNet architecture. I tried adding new layers to 
gradually reduce the number of pixel dimensions to 32x32 for the standard LeNet, and then sticking to the stock LeNet 
architecture from there on. 

I also tried different image resize operations to keep the architecture to a basic LeNet. 

Another approach I tried was to preprocess the image using various approaches, employing preprocessing Lambda layers 
in my Keras model and also using the ImageDataGenerator class of Keras.

However, I was not able to produce a successful model with any of these approaches. 

It should be noted that at various points, after discovering the models I tried were struggling at specific portions 
of the track, I recorded more training data, on some occasions completely from scratch. I tried to very carefully 
collect correcting movements both near the center line and closer to the edges of the road. 
 
As a conclusion to my attempts with these simpler models, I abandoned the (modified) LeNet approach and decided to 
try something else. 

##### Feature extraction

There was also a point where I considered using the bottleneck features from the previous lectures. However, in my 
initial attempts to draft a model and training framework for feature extraction, I was not able to put together all the 
necessary pieces using Keras. It would probably have been easier using TensorFlow, but I recognized the requirement 
that the model be implemented in Keras so I abandoned the feature extraction route. 

###### Review of the lecture material

Looking for new ideas, I resorted to the lecture material and finally discovered the Nvidia model. Straight away after 
giving it a shot, I noticed it performed significantly better than my previous attempts. 
  
From this point on, I decided to stick to the Nvidia architecture and find a version of it that would successfully 
drive the track. 

#### 

In order to gauge how well the model was working, I split my image and steering angle data into a training and 
validation set. I found in my early experiments that the learning was very sensitive on the training data and presense 
of carefully chosen dropout layers etc. Sometimes the model had a low mean squared error on the training set but a 
high mean squared error on the validation set which is a sign of overfitting. On other occasions, also the training 
set MSE would not get low enough, probably a sign of either poor training data, poor network architecture, or poorly chosen 
network parameters. Gradually, I learnt to monitor the metrics to achieve a fast enough drop in the loss metrics of both 
the training and validation set, and based on these observations, I was also able to choose suitable dropout rate and 
epoch count. 

The final step was to run the simulator to see how well the car was driving around track one. At various points during 
the work, there were a spots where the vehicle fell off the track. Some bits were particularly challenging. The 
evolution of my network architecture and the different preprocessing and normalization techniques I tried is 
documented above, and finally I was able to construct a solution that would enable the car to stay on track. 

#### 2. Final Model Architecture

The final model architecture (model.py lines FIXME 18-24), documented briefly above, is illustrated in the following 
image, reproduced from the Nvidia blog: 

![An image of the architecture I used, based on the Nvidia solution][nvidia_architecture]

As can be seen, I have slightly modified the network by changing the number of neurons in the fully connected layers
and introducing dropout layers after all the dense layers and also the flattening layer before the fully connected part
of the architecture.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first tried recording mostly center lane driving. Here is an example image of 
center lane driving: 

![Driving in the center][center_lane_driving]

During my initial attempts at training the model, I came to the conclusion that I had recorded too much of redundant 
center line driving behavior, and the parts of the track where the vehicle was not behaving well needed to be 
emphasized in the training material. Subsequently, I included more of correcting behavior and portions of the track 
where the edges or the texture of the lane were different or changing.  

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that 
the vehicle would learn to return the center after deviating towards the edges. 
In this process, I figured it would be detrimental to the model if the movements that lead toward the edges  
were included in the training set, so I turned recording off while approaching the edges, then turned the tires toward 
the center of the track and only then start recording again. 

These images show what a recovery looks like starting from the edges:

![Recovering from right][correction_from_right]
![Recovering from left][correction_from_left]
![Recovering from a differently textured edge][correcting_from_gray_edge]

The recovery portions were recorded throughout the track. Here I wanted to include images from 
parts of the track where also the texture or surroundings of the track varied a bit from what was
mostly seen.

I then also tried repeating this process on track two in order to get more data points. However, in my experiments, 
this did not improve the model so I ended up not using the data from track two. 

To augment the data sat, I also flipped images and angles thinking that this would lead to better generalization.  
For example, here is an image that has then been flipped:

![An image that has been flipped][flipped_image]

Of course, when flipping the images, I also inverted the steering angles. Otherwise the training data would  
be nonsensical. 

After the collection process, I had FIXME: number of data points. I then preprocessed this data by scaling and centering the 
rgb values to the -0.5..0.5 range by dividing by 255 and subtracting 0.5. Another preprocessing step was to crop the 
images, omitting 70 pixels from the top and 20 from the bottom, as well as 8 pixels from each side (see 
code line 52).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under 
fitting. The ideal number of epochs was 5 as evidenced by a careful monitoring of the training and validation losses. 
Below is the record of a typical run of the model using too many epochs. As can be seen in the data, after 5 epochs, 
the training loss continues to drop. However, the drop is not really significant, especially taking into account that 
validaiton loss starts to increase.

I used an adam optimizer so that manually training the learning rate wasn't necessary.


