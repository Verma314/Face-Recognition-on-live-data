import numpy as np
import cv2
import tensorflow as tf, sys



####### av:				initialize openCV stuff 
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
ret, img = cam.read()
id = raw_input("Enter user id : ")
#######






######## av: 				initialize tensorflow classifier

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/home/verma314/image_classification/output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/home/verma314/image_classification/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


#now all we need to do is give tensorflow the image data
##first lets create the image data and then we shall use the tensorflow code to run it on the dataset


sampleNum = 0
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,
                                          scaleFactor=1.3,
                                          minNeighbors=5
                                          )

    for (x,y,w,h) in faces:
        sampleNum = sampleNum+1
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100);
        
    cv2.imshow('face',img);
    cv2.waitKey(1);
    if sampleNum > 20:
        break
cam.release()
cv2.destroyAllWindows()


#the images have been stores in dataSet, now we shall get all the images and do a classification


#todo: import images from the dataSet file




##load the images into image_path, the classifier has to run 20 times:
i = 1
while ( i <= 20):	
	#gets the images one by one	
	image_path = "dataSet/User."+str(id)+"."+str(i)+".jpg"
	image_data = tf.gfile.FastGFile(image_path, 'rb').read()
	
	with tf.Session() as sess:
    		# Feed the image_data as input to the graph and get first prediction
    		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    		predictions = sess.run(softmax_tensor, \
             		{'DecodeJpeg/contents:0': image_data})
    
    		# Sort to show labels of first prediction in order of confidence
    		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    		for node_id in top_k:
        		human_string = label_lines[node_id]
        		score = predictions[0][node_id]
        		print('%s (score = %.5f)' % (human_string, score))
	i = i +1


























