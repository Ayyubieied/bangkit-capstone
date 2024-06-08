from django.urls import path
from machinelearningbackend.views import SkinMetrics

app_name = 'machinelearningbackend' 

urlpatterns = [
    path('upload/', SkinMetrics.as_view(), name='upload'),
]