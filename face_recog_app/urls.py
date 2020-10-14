from django.urls import path, include
from . import views
urlpatterns = [
 path("recog_face/", views.recog_face, name = "recog_face"),
 path("recognize_image/", views.RecognizeImage.as_view(), name="recognize_image"),
 path("person_id_image/", views.PersonIdImageClass.as_view(), name="person_id_image"),
  
]