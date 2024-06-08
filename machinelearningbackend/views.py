import os
import base64
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from .serializers import ImageUploadSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.base import ContentFile
from tempfile import NamedTemporaryFile
from .models.skin_tone.skin_tone_knn import identify_skin_tone

class_names1 = ['Dry_skin', 'Normal_skin', 'Oil_skin']
class_names2 = ['Low', 'Moderate', 'Severe']
skin_tone_dataset = 'models/skin_tone/skin_tone_dataset.csv'

model1 = load_model('./models/skin_model')
model2 = load_model('./models/acne_model')

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

def prediction_skin(img_path):
    new_image = load_image(img_path)
    pred1 = model1.predict(new_image)
    if len(pred1[0]) > 1:
        pred_class1 = class_names1[tf.argmax(pred1[0])]
    else:
        pred_class1 = class_names1[int(tf.round(pred1[0]))]
    return pred_class1

def prediction_acne(img_path):
    new_image = load_image(img_path)
    pred2 = model2.predict(new_image)
    if len(pred2[0]) > 1:
        pred_class2 = class_names2[tf.argmax(pred2[0])]
    else:
        pred_class2 = class_names2[int(tf.round(pred2[0]))]
    return pred_class2

class SkinMetrics(APIView):
    def post(self, request, format=None):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            img_temp = NamedTemporaryFile(delete=True)
            img_temp.write(image.read())
            img_temp.flush()
            skin_type = prediction_skin(img_temp.name)
            acne_type = prediction_acne(img_temp.name)
            tone = identify_skin_tone(img_temp.name, dataset=skin_tone_dataset)
            return Response({'type': skin_type, 'tone': str(tone), 'acne': acne_type}, status=200)
        return Response(serializer.errors, status=400)