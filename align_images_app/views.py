from django.shortcuts import render
from lib.src.align.align_dataset_mtcnn import main
from tcs_face_recog import settings
import os
# Create your views here.
def align_images_from_unalign_datab(request):
	#dir having unaligned images
	input_dir=os.path.join(settings.BASE_DIR , f'media/unaligned_datab')
	#aligned images will be storein this dir
	output_dir=os.path.join(settings.BASE_DIR , f'media/datab')
	image_size=160
	margin=44
	random_order=False
	gpu_memory_fraction=1.0
	detect_multiple_faces=False
    #we just need to pass important paraeters that is input dir and output dir to this function
    #this function will take all images from input dir and align one by one and save it to output_dir
    #note...dir structure of input_dir and out_put dir will be same after execution of below function
	main(input_dir,output_dir,image_size,margin,random_order,gpu_memory_fraction,detect_multiple_faces)
	return render(request, "successful_images_align_in_datab.html")

