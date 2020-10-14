from django.urls import path, include
from . import views
urlpatterns = [
  path("align_images_from_unalign_datab/", views.align_images_from_unalign_datab, name="align_images_from_unalign_datab"),

]