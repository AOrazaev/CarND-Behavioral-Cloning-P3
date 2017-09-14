# **Behavioral Cloning** 

## Self-driving car Udacity project


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[keras_summary]: ./examples/keras_summary.png "Keras summary"
[preprocessing]: ./examples/preprocessing.png "Preprocessing steps"
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [DriveNet.ipynb](./DriveNet.ipynb) jupyter notebook containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [drivenet.h5](./drivenet.h5) containing a trained convolution neural network 
* [writeup_template.md](./README.md) summarizing the results
* [video.mp4](./video.mp4) video demonstration of results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py drivenet.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

To extract visual features from pictures was decided to use MobileNet network pre-trained on ImageNet dataset without fully-connected layers.

MobileNet shows comparable results on the ImageNet dataset. And has much smaller size than InceptionV3 or ResNet50. This also results in faster forward propagation.

Normalization is built-in in final the final neural network as a Lambda layer before MobileNet pass.

There is 3 fully-connected layers after visual features extraction with sizes: 256, 128, 1.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers after first 2 fully-connected layers.

The model was trained, validated and tested on different data sets to ensure that the model was not overfitting. At the end the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road...

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to chose appropriate pre-trained neural network to extract visual features. Fine tune it.

At the beginning I tried to use ResNet50, but I wasn't happy with training time, so I decided to move to faster model with comparable capabilities. MobileNet was chosen.

Because MobileNet expects squared input there is additional steps applied. At first, he image is rescaled to size 160x320 and after this crop it to size 160x160 throwing away noize from sky and car's front part:

![preprocessing][preprocessing]

As a result we throw away noize and rescale image to MobileNet expected input size.

After first training results were poor, data collected by me was small and bad quality because I collected using keyboard and mouse. After this I found Xbox360 controller and collected better data with smoother steering angle change and more precise driving. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track: bridge with different texture and turn after the bridge. To improve the driving behavior in these cases, I collected more data from this problematic spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

![Keras summary][keras_summary]

[Here is the link](https://arxiv.org/pdf/1704.04861.pdf) to MobileNet paper for details about base network.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to center of the road.

To augment the data set, I also flipped images. I decided not to use data from left and right cameras, because I didn't want to apply any additional heuristic (which possibly can be wrong) on the steering angle change.

After the collection process, I had 29088 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set and 10% into test set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Final MSE loss is:
train: 0.0076
validation: 0.0118
test: 0.0118
