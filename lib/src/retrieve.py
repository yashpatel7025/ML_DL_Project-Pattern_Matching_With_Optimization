
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
from .facenet import load_data,load_img,load_model,to_rgb
from . import lfw
import os
import sys
import math

from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import cv2

from .align import detect_face
from datetime import datetime
from scipy import ndimage
from scipy.misc import imsave 
from scipy.spatial.distance import cosine
import pickle
from oauth2client import tools
#face_cascade = cv2.CascadeClassifier('out/face/haarcascade_frontalface_default.xml')
from face_recog_app.models import *
from imgs_upload.models import *
import re
from tcs_face_recog import settings
import string 
import random 
import imutils

    
image_size=160
detect_multiple_faces=True
margin=44

def align_face(img,pnet, rnet, onet):
                global detect_multiple_faces,margin

                minsize = 20 # minimum size of face
                threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
                factor = 0.709 # scale factor

                 
                if img.size == 0:
                    return False,img,[0,0,0,0]

                if img.ndim<2:
                    print('Unable to align')

                if img.ndim == 2:
                    img = to_rgb(img)

                img = img[:,:,0:3]
    
                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

                nrof_faces = bounding_boxes.shape[0]

                        
                if nrof_faces==0:
                    return False,img,[0,0,0,0]
                else:
                    det = bounding_boxes[:,0:4]
                    det_arr = []
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces>1:
                        if detect_multiple_faces:
                            for i in range(nrof_faces):
                                det_arr.append(np.squeeze(det[i]))
                        else:
                            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                            img_center = img_size / 2
                            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                            det_arr.append(det[index,:])
                    else:
                        det_arr.append(np.squeeze(det))
                    if len(det_arr)>0:
                        faces = []
                        bboxes = []
                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-margin/2, 0)
                            bb[1] = np.maximum(det[1]-margin/2, 0)
                            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                            # misc.imsave("cropped.png", scaled)
                            faces.append(scaled)
                            bboxes.append(bb)
                            print("leaving align face")
                        return True,faces,bboxes
            

def identify_person(image_vector, feature_array, k=9):
        top_k_ind = np.argsort([np.linalg.norm(image_vector-pred_row) \
                            for ith_row, pred_row in enumerate(feature_array.values())])[:k]
       
        result = list(feature_array.keys())[top_k_ind[0]]
        acc = np.linalg.norm(image_vector-list(feature_array.values())[top_k_ind[0]])
        return result, acc



def recognize_face(sess,pnet, rnet, onet,feature_array):
    # Get input and output tensors
    global image_size
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    
    #images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    image_size = image_size
    embedding_size = embeddings.get_shape()[1]

    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, 0)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if(gray.size > 0):
            
            gray=imutils.resize(gray, width=1000)
            response, faces,bboxs = align_face(gray,pnet, rnet, onet)
            # print(response)
            # print(faces)
            # print(bboxs)
            if (response == True):
                    for i, image in enumerate(faces):
                            bb = bboxs[i]
                            
                            # cv2.imshow('ImageWindow', image)
                            # cv2.waitKey()
                            
                            images = load_img(image, False, False, image_size)

                            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                            feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                            result, accuracy = identify_person(feature_vector, feature_array,8)
                            print()
                            print("accuracy:- ",accuracy)
                            print("result:- ",result)
                            #print(result.split("/")[2])
                            

                            if accuracy < 0.85:
                            
                                cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                                W = int(bb[2]-bb[0])//2
                                H = int(bb[3]-bb[1])//2
                                basename=os.path.basename(os.path.normpath(result))
                                cv2.putText(gray,"Hello "+basename,(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                            else:
                                cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                                W = int(bb[2]-bb[0])//2
                                H = int(bb[3]-bb[1])//2
                                cv2.putText(gray,"WHO ARE YOU ?",(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                            del feature_vector

            cv2.imshow('img',gray)
        else:
            continue

def my_func_for_recognize_face_in_1_image_only(img_url,sess,pnet, rnet, onet,feature_array):
    # Get input and output tensors
    global image_size
    images_placeholder = sess.graph.get_tensor_by_name("input:0")

    #images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    image_size = image_size
    embedding_size = embeddings.get_shape()[1]

    cap = cv2.VideoCapture(img_url)
    
    flag=1
    while(flag):
        flag=0
        ret, frame = cap.read()

        if ret:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            return 0,0,0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if(gray.size > 0):
            
            response, faces,bboxs = align_face(gray,pnet, rnet, onet)
            
            if (response == True):
                    for i, image in enumerate(faces):
                            bb = bboxs[i]
                            
                            # cv2.imshow('ImageWindow', image)
                            # cv2.waitKey()
                            images = load_img(image, False, False, image_size)
                            
                            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                            feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                            result, accuracy = identify_person(feature_vector, feature_array,8)
                            print()
                            print('aaakakakakakakakakakkakakakakakakkak')
                            print("accuracy:- ",accuracy)
                            print("result:- ",result)
                            
                            

                            if accuracy < 0.9:
                            
                                cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                                W = int(bb[2]-bb[0])//2
                                H = int(bb[3]-bb[1])//2
                                basename=os.path.basename(os.path.normpath(result))
                                cv2.putText(gray,"Hello "+basename,(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                                

                                basename=os.path.basename(os.path.normpath(result))
                                print("basename ", basename)
                                folder_name = " ".join(re.findall("[a-zA-Z]+", basename))
                                extension= folder_name.split()[1]
                                folder_name=folder_name.split()[0]


                            else:
                                cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                                W = int(bb[2]-bb[0])//2
                                H = int(bb[3]-bb[1])//2
                                cv2.putText(gray,"WHO ARE YOU ?",(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                                


                                
                                folder_name = "unknown"
                                extension= 'jpg'
                                



                            del feature_vector


                            
                            random_str = str(''.join(random.choices(string.ascii_uppercase +string.digits, k = 7)) )

                            print("folder_name ",folder_name) 
                            p_obj=People.objects.filter(name=folder_name)
                            if p_obj.exists():
                                 people_obj = p_obj.first()
                                 rec_obj = RecogedImageUploadedImage(name=folder_name)
                            else:#unknown
                                 rec_obj = RecogedImageUploadedImage(name=f"unknown_{random_str}")
                                 people_obj = 0
                                 
                            

                            
                            file_name=folder_name+random_str+'.'+extension
                            
                        


                            path = os.path.join(settings.BASE_DIR, f'media/predicted_images_upload_to_predict/{folder_name}') 
                            if not os.path.exists(path):
                                os.mkdir(path) 

                            img_path=os.path.join(settings.BASE_DIR , f'media/predicted_images_upload_to_predict/{folder_name}/{file_name}')
                            #Imwrite() only save image if dir is present

                            cv2.imwrite(img_path, gray) 

                            
                            # img_obj=Predicted(image_url=path_to_save_img,people=people_obj)
                            # img_obj.save()
                            rec_obj.image_url = 'http://127.0.0.1:8000/'+f'media/predicted_images_upload_to_predict/{folder_name}/{file_name}'
                            rec_obj.save()

                            if people_obj:
                                id_img = PersonIdImage.objects.get(people=people_obj).image.url
                                return 'http://127.0.0.1:8000/'+f'media/predicted_images_upload_to_predict/{folder_name}/{file_name}',people_obj.id,id_img
                            else:
                                return 'http://127.0.0.1:8000/'+f'media/predicted_images_upload_to_predict/{folder_name}/{file_name}',0,0
                            
                            
            else:
                return 0,0,0
            # cv2.imshow('img',gray)
        else:
            return 0,0,0