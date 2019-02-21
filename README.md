# Face Recognition on Live Video Streams


**Abstract— Implementing a system to detect faces in videos and to identify the person using a trained deep neural network.
Keywords—face recognition; tensorflow; deep learning; transfer learning; parallel programming**

### I.  Introduction 

With the emergence of tools such as TensorFlow and CUDA it has become easier to solve progressively difficult problems in areas such as machine learning, deep learning, computer vision and so on.  A lot of open problems have been solved in recent years in image processing. Hence we turn to video processing, which is still an unsaturated field.
We use open source tools openCV and TensorFlow to re-train the Inception v3 model to classify human faces in a live video stream.

###  II.  Objective

Our aim is to re-train an neural networks model, with our own faces.  We want to detect faces in live streamed videos and recognize the person. 
     We re-train the final layer of Inception v3 with cropped images of human faces. We then use the model to classify faces detected in a video stream.

### III. Methodology

A. Dataset Creation

To create the dataset, 200 images of each person to be trained is collected using the same live streaming code that is used in face detection [1].  The image is converted to gray-scale so that the background color does not affect the image and the features of the person. It also ensures that the size of all the cropped images of the face is the same.

Face detection from the video is done in python using the another pre-trained model “haarcascade_frontalface_default”, which detects the frontal face [1].



B. Training 

Training a completely new neural network for face recognition may be cumbersome and take a lot of time and processing to fully train.  We use  a concept Transfer Learning to achieve the objective.  It is a technique in which we take a fully trained model and retrain from the existing weights for new classes [3].

We take the Inception v3 model, which has been trained on ImageNet dataset.  We run a script called retrain.py from the open source project TensorFlow.  Once we have the dataset created in the previous section, the script starts to train the top layer of the network.  The script will run 4000 training steps, giving a series of outputs, each showing training accuracy, validation accuracy and cross entropy. Each step chooses ten images at random from the training set, finds their bottlenecks from the cache, and feeds them into the final layer to get predictions. Those predictions are then compared against the actual labels to update the final layer's weights through the back-propagation process. [2]

It writes out a modified version of the Inception v3 network as output_graph.pb and the set of the custom labels as a text file.

C. Recognition
We then write a script to take live feed of images from the web camera and use the retrained model on those images. 

### IV. Conclusion

The training accuracy tells us the number of images from the training batch that were labeled as the correct class. Validation accuracy is the precision on a randomly image from a set the mode has not seen before. Cross entropy tells us how well the learning process is progressing. [3]

These parameters are run on each of the 4000 training steps.  We can see the accuracy improve in each step and go up to greater than 99.0%.

After the training, a final test for accuracy is done, which gives us an an accuracy between 90%- 95%.
We then used the re-trained Inception v3 model to classify images fed directly through the web camera.

### V. Applications

a) Security: Face recognition from streamed data can be used to detect and restrict entry of unauthorized people into the premises. Moreover, the neural nets can be trained to detect the whether two wheeler drivers have helmets or not.
It can also be used to grant access to certain people and can be used in personal computers to grant access.

b) Biometric attendance:  We can train the neural network to identify people from a whole class, and use the face recognition system to create an automated biometric attendance system.

c) Statistics:  The software can be altered  to count the number of people present in a certain area.


### VI. Future Scope

The identification of a person can be done in parallel for faster and real-time processing.

In our implementation, we have used CUDA and parallel programming only for the training phase, it can be extended to the testing phase as well.


### References


[1] OpenCV documentation, cascade-classifier http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html#cascade-classifier

[2] TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems by Martin Abadi, Ashish Agarwal et al.

[3] TensorFlow Documentation, Image Retraining https://www.tensorflow.org/tutorials/image_retraining

[4] Made using Creately: https://creately.com/
