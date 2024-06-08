from django.urls import path
from .views import SkinMetrics

urlpatterns = [
    path('upload/', SkinMetrics.as_view(), name='upload'),
]