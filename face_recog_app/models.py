from django.db import models
from imgs_upload.models import People
# Create your models here.
def user_directory_path_for_to_predict(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    #return 'user_{0}/{1}'.format(instance.user.id, filename)
    #instance would be image object
    return 'image_upload_to_predict' +'/'+filename

class ImageUploadToRecog(models.Model):
	image=models.FileField(upload_to =user_directory_path_for_to_predict)
	def __str__(self):
		return f' {self.id }'+'_image'

class RecogedImageUploadedImage(models.Model):

    image_url=models.URLField(max_length=1000,null=True,blank=True)
    name = models.CharField("name", max_length=200, null=False, blank=False)
    
    def __str__(self):
        return f' {self.name}'+'_image'
	
class LiveRecognizedImage(models.Model):
    
    image=models.FileField(upload_to ="RecognizesImages")
    people=models.ForeignKey(People,on_delete=models.CASCADE)

    def __str__(self):
        return f' {self.people.name }_{self.id}'+'image'
