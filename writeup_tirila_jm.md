#**Behavioral Cloning** 

##Writeup, Juha-Matti Tirilä 


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
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted and Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by 
executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the 
pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with first 5x5 and subsequently also 3x3 filter sizes and depths 
between 24 and 64 (model.py lines FIXME: 18-24). These convolutional layers are followed by fully connected layers, 
starting with one containing 100 neurons, and in the subsequent layers the neuron count is gradually reduced to 1 
to get scalar output. 

The model includes RELU layers to introduce nonlinearity (code lines FIXME), and the data is normalized in the model using 
a Keras lambda layer (code line FIXME). 

####2. Reducing overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines FIXME 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 
FIXME 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on 
the track.

####3. Model parameter tuning

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

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and 
recovering from both sides of the road. The recovery actions were carefully recorded in order to just capture the 
corrective steering, not the erraneous actions before that.

###Model Architecture and Training Strategy

####1. Solution Design Approach

##### AlexNet

The overall strategy for deriving a model architecture was to first try experimenting with some of the advanced 
architectures like Alexnet.  

However, I soon run into problems with system resources also on AWS. Meanwhile, some participants reported they had 
trained models successfully using quite simple architectures so 
I also decided to let go of the idea of advanced models. 
  
##### Back to Basics  

I then decided to start with something simple, try different approaches and was hoping to finally settle on something 
that drives the track successfully. 
  
As a next step, kind of to achieve a reference level, I designed a simple multilayer perceptron model. 

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

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers 
and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project 
rubric)

![An image of the architecture I used, based on the Nvidia solution][nvidia_architecture]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded some center lane driving. Here is an example image of center lane 
driving:

During my initial attempts at training the model, I came to the conclusion that I had recorded too much of redundant 
center line driving behavior, and the parts of the track where the vehicle was not behaving well needed to be 
emphasized in the training material. Subsequently, I included more of correcting behavior and portions of the track 
where the edges or the texture of the lane were different or changing.  

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that 
the vehicle would learn to return the center after deviating towards the edges. 
In this process, I figured it would be detrimental to the model if the movements that lead toward the edges  
.... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

I then I also repeated this process on track two in order to get more data points. However, in my experiments, 
this did not improve the model so I ended up not using the data from track two. 

To augment the data sat, I also flipped images and angles thinking that this would lead to better generalization.  
For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had X number of data points. I then preprocessed this data by scaling and centering the 
rgb values to the -0.5..0.5 range by dividing by 255 and subtracting 0.5. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under 
fitting. The ideal number of epochs was 5 as evidenced by a careful monitoring of the training and validation losses. 
Below is the record of a typical run of the model using too many epochs. As can be seen in the data, after 5 epochs, 
the training loss continues to drop. However, the drop is not really significant, especially taking into account that 
validaiton loss starts to increase.


I used an adam optimizer so that manually training the learning rate wasn't necessary.
