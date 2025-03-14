from django.urls import path
from .views import file_upload_view

urlpatterns = [
    path('', file_upload_view, name="file_upload"),
]
