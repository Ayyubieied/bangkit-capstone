from django.urls import path
from machinelearningbackend.views import *

app_name = 'machinelearningbackend' 

urlpatterns = [
    path('upload/', skin_metrics, name='upload'),
]