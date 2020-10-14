

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from django.shortcuts import render
import tensorflow as tf
import numpy as np
import argparse
from lib.src import facenet
#import lfw
import os
import sys
import math
#import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import numpy as np
import pickle

graph = tf.get_default_graph()
def main(model,image_size):
    global graph
  
    with graph.as_default():# Create a new graph, and make it the default.
      
        with tf.Session(graph=graph) as sess:# `sess` will use the new, currently empty, graph.
        # Build graph and execute nodes in here.
            
            paths=[]
            datab_path= "media/datab/"
            for img_class_name in os.listdir(datab_path):
                all_imgs_of_img_class_name =[datab_path + img_class_name+'\\' + f for f in os.listdir(datab_path + img_class_name)]
                paths.extend(all_imgs_of_img_class_name)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
   
            # images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
            # print("images_placeholder.................",images_placeholder)
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            embedding_size = embeddings.get_shape()[1]
            
            face_embeddings_dict = {}
            
            # Run forward pass to calculate embeddings
            for i, filename in enumerate(paths):
                print("i..",i)
                print("filename.....",filename)
                #load_image() will give us array form of image
                images = facenet.load_image(filename, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                basename=os.path.basename(os.path.normpath(filename))
                face_embeddings_dict[basename] =  feature_vector
                print(feature_vector.shape)
                if(i%100 == 0):
                    print("completed",i," images")
            print()

            #save as pickle file for later use
            with open('create_face_embeddings_app/face_embeddings_dict.pickle','wb') as f:
                pickle.dump(face_embeddings_dict,f)

        

def create_face_embeddings(request):
    #Our trained model path:-- facenet model
    model='trained_models/ckpt/ckpt20180408-102900.pb'
    image_size=160
    #this function is written above ...it will take all images one by one from each person folder 
    #and each image will be given to our trained model ...for each image it will give us corresponding embedding...
    #each embedding we are storing it in dictionary with filename as key and embeding as value
    #after getting all embeddings of all images will store dictionary in pickle file for later use. 
    main(model,image_size)
    
    return render(request, 'successful_embedding_save.html' )


