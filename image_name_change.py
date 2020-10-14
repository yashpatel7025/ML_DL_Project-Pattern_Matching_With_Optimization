#after saving renamed images i think size of image maximize 
# Python program to explain cv2.imwrite() method 

# importing cv2 
import cv2 

# importing os module 
import os 

datab_path="C:\\Users\\Aakash\\Desktop\\tcs_face_recognition\\media\\upload_image_from_this_folder\\"
for img_class_name in os.listdir(datab_path):
	all_imgs_of_img_class_name =[datab_path + img_class_name+'\\' + f for f in os.listdir(datab_path + img_class_name)]
	output_directory = "C:\\Users\\Aakash\\Desktop\\tcs_face_recognition\\media\\tp\\" + img_class_name 
	temp=1
	os.mkdir(output_directory)
	for image_path in all_imgs_of_img_class_name:
		# Using cv2.imread() method 
		# to read the image 
		img = cv2.imread(image_path) 

		# Change the current directory 
		# to specified directory 
							
		os.chdir(output_directory) 

		# Filename 
		filename = img_class_name+ str(temp)+'.jpg'
		temp=temp+1

		# Using cv2.imwrite() method 
		# Saving the image 
		cv2.imwrite(filename, img) 

	print(f'Successfully saved {img_class_name}') 
