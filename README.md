- Go to tcs_face_recognition directory
- run command to make virtual env--->virtualenv ve
- activate virtual env --> ve\scripts\activate
- install following packages by command--->pip install -r requirements.txt
- run cmd---> python manage.py runserver
- open browser go to url-->http://127.0.0.1:8000/
- enter username and pwd----test1234,test@1234
- you will be landing on home page of web app


# folder explaination:

1. align_images_app : 

in this folder we have all the codes related to aligning image i.e we want just face image from whole person image..

2. create_face_embeddings_app :

in this folder we have all the codes related to creating pickle file which will contain dictionary of our person images and their corresponding image embedding predicted by our 
facenet pretrained model

3. imgs_upload

in this folder we have all codes related to upload images of each person in database

4. face_recog_app

in this folder we have codes related to live recognization and recognizing image which is uploaded

5. lib

in lib folder we have imp below sub files
a.facenet.py :- contains code for loading image,loading model...
b.retrieve.py :- contains code for recognizing face and identifying person by calculating difference between embeddings
c.detect_face.py :- contains code for detecting face using mtcnn algo
d.align_dataset_mtcnn.py :- code for detecting face and saving it in datab folder

6. media

contains all images for recognization,which arerecognized,align images,unalign images
etc etc

7. templates

contains all html files

8. trained models

having our trained facenet model
------------------------------------------

for more detail explaination of the code and what each folder contains plz watch video i have send which explains everything in detail ..
-----------------------------
Demo videos

https://drive.google.com/open?id=1XQy2wa2gEQ4EJX93Kq5ERQNCr__-Yn1H

-------------------------------

For any issues kindly contact
Yash Patel
7021875166
yashpatel7025@gmail.com
----------------------------