from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register(ImageUploadToRecog)
admin.site.register(RecogedImageUploadedImage)
admin.site.register(LiveRecognizedImage)