from django.urls import path, include
from . import views
urlpatterns = [
  path("create_face_embeddings/", views.create_face_embeddings, name="create_face_embeddings"),
]