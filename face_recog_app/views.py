from django.shortcuts import render,redirect

# Create your views here.
import os
import sys
import random
from tensorflow.python.platform import gfile
from six import iteritems

from lib.src import facenet 

import numpy as np
from lib.src import retrieve
from lib.src.align import detect_face
import tensorflow as tf
import pickle
from tensorflow.python.platform import gfile

from django.views.generic.edit import FormView
from .forms import *
import cv2
from imgs_upload.models import *
from .models import *
from tcs_face_recog import settings
 from django.contrib import messages


try:#if file not exit at beginning before performing any face_embedding
    with open(os.path.join(settings.BASE_DIR , f'create_face_embeddings_app/face_embeddings_dict.pickle'),'rb') as f:
        feature_array = pickle.load(f) 
except:
    pass


model_exp = os.path.join(settings.BASE_DIR , f'trained_models/ckpt/20180408-102900.pb')
#load trained facenet model..
facenet.load_model(model_exp)

graph_fr = tf.get_default_graph()


sess_fr = tf.Session(graph=graph_fr)
with graph_fr.as_default():
    
    #this line will take time to execute
    pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)

def recog_face(request):
    #live recognition will start...all code is written in retrieve file
    retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array)

    return redirect('/')

#logic for recognizing uploaded image
class RecognizeImage(FormView):
    form_class = FileFieldForm_2
    template_name = 'recognize_image.html'  # Replace with your template.
    success_url = '/images_upload'  #doesnot matter successurl here coz in form_valid() we redirceing to after_recog.html
    context= "not_now"
    
    def get(self, request, *args, **kwargs):
        
        form = self.form_class()
        return render(request, self.template_name, {'form': form})


    def post(self, request, *args, **kwargs):
    
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file_field')
        
        if form.is_valid():

            
            frame = cv2.imdecode(np.fromstring(request.FILES['file_field'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

            image_to_recognize = files[0]
            

            img_obj=ImageUploadToRecog(image=image_to_recognize)
            img_obj.save()

            
            img_url=settings.BASE_DIR + '/media/image_upload_to_predict/' + str(image_to_recognize)
        
            #recognize image ...save recognized image in database
            gray_img_url, people_obj_id, id_img_url = retrieve.my_func_for_recognize_face_in_1_image_only(img_url,sess_fr,pnet, rnet, onet,feature_array)
            if gray_img_url==0 and people_obj_id==0:
                return render(self.request,"invalid_image.html")
            elif gray_img_url != 0 and  people_obj_id == 0:
                
                self.context={
                   'predicted_person' :"unknown",
                   'predicted_person_id' : "unknown",
            
                    'uploaded_image_url' : gray_img_url ,
                    'predicted_person_url':'/media/unknown.png'
                    }
            else:
                people_obj=People.objects.get(id=people_obj_id)
                self.context={
                     'predicted_person' : people_obj.name,
                    'predicted_person_id' : people_obj.id,
            
                     'uploaded_image_url' : gray_img_url,
                     'predicted_person_url':id_img_url
                   }

            return self.form_valid(form)#if form is valid
        else:
            return self.form_invalid(form)
    
    def form_valid(self, form):
        # We make sure to call the parent's form_valid() method because
        # it might do some processing (in the case of CreateView, it will
        # call form.save() for example).
        response = super().form_valid(form)
        return render(self.request,"after_recognize_image.html",self.context)

#logic for uploading unique person image as an id of person
class PersonIdImageClass(FormView):
    form_class = FileFieldForm_2
    template_name = 'add_unique_image.html'  
    success_url = '/'  
    context= "not_now"
   
    
    def get(self, request, *args, **kwargs):
        
        form = self.form_class()
        return render(request, self.template_name, {'form': form})


    def post(self, request, *args, **kwargs):
       
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file_field')
        
        if form.is_valid():
            image = files[0]
            person_id=int(request.POST['person_id'])
            p_obj = People.objects.get(id=person_id)
            p_id_obj=PersonIdImage(people=p_obj,image=image)
            p_id_obj.save()
            messages.add_message(self.request, messages.INFO, f"Successfully Uploaded Id image")
            return self.form_valid(form)#if form is valid
        else:
            return self.form_invalid(form)
    
    
        