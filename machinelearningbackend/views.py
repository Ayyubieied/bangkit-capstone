import os
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from machinelearningbackend.models.skin_tone.skin_tone_knn import identify_skin_tone
from .serializers import ImageUploadSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.base import ContentFile
from tempfile import NamedTemporaryFile
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

class_names1 = ['Dry_skin', 'Normal_skin', 'Oil_skin']
class_names2 = ['Low', 'Moderate', 'Severe']
skin_tone_dataset = 'machinelearningbackend/models/skin_tone/skin_tone_dataset.csv'

model1 = load_model('machinelearningbackend/models/skin_model')
model2 = load_model('machinelearningbackend/models/acne_model')

df2 = pd.read_csv('machinelearningbackend/models/recommender/final.csv')
makeup = pd.read_csv('machinelearningbackend/models/recommender/makeup_final.csv')
entries = len(df2)
LABELS = list(df2.label.unique())

features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive', 'fine lines', 'wrinkles', 'redness',
            'dull', 'pore', 'pigmentation', 'blackheads', 'whiteheads', 'blemishes', 'dark circles', 'eye bags', 'dark spots']

def search_concern(target, i):
    if target in df2.iloc[i]['concern']:
        return True
    return False


def name2index(name):
    return df2[df2["name"] == name].index.tolist()[0]


def index2prod(index):
    return df2.iloc[index]

def convert_to_rupiah(price_in_rupees):
    price_in_rupees = price_in_rupees.replace("₹", "").strip()
    price_in_rupiah = int(price_in_rupees) * 195
    return "Rp " + str(price_in_rupiah)


def wrap(info_arr):
    result = {}
    result['brand'] = info_arr[0]
    result['name'] = info_arr[1]
    result['price'] = convert_to_rupiah(info_arr[2])
    result['url'] = info_arr[3]
    result['img'] = info_arr[4]
    result['skin type'] = info_arr[5]
    result['concern'] = str(info_arr[6]).split(',')
    return result

def wrap_makeup(info_arr):
    result = {}
    result['brand'] = info_arr[0]
    result['name'] = info_arr[1]
    result['price'] = info_arr[2]
    result['url'] = info_arr[3]
    result['img'] = info_arr[4]
    result['skin type'] = info_arr[5]
    result['skin tone'] = info_arr[6]
    return result

one_hot_encodings = np.zeros([entries, len(features)])


for i in range(entries):
    for j in range(5):
        target = features[j]
        sk_type = df2.iloc[i]['skin type']
        if sk_type == 'all':
            one_hot_encodings[i][0:5] = 1
        elif target == sk_type:
            one_hot_encodings[i][j] = 1

for i in range(entries):
    for j in range(5, len(features)):
        feature = features[j]
        if feature in df2.iloc[i]['concern']:
            one_hot_encodings[i][j] = 1


def recs_cs(vector = None, name = None, label = None, count = 5):
    products = []
    if name:
        idx = name2index(name)
        fv = one_hot_encodings[idx]
    elif vector:
        fv = vector
    cs_values = cosine_similarity(np.array([fv, ]), one_hot_encodings)
    df2['cs'] = cs_values[0]
    
    if label:
        dff = df2[df2['label'] == label]
    else:
        dff = df2
    
    if name:
        dff = dff[dff['name'] != name]
    recommendations = dff.sort_values('cs', ascending=False).head(count)
    data = recommendations[['brand', 'name', 'price', 'url','img','skin type','concern']].to_dict('split')['data']
    for element in data:
        products.append(wrap(element))
    return products

def recs_essentials(vector = None, name = None):
    response = {}
    for label in LABELS:
        if name: 
            r = recs_cs(None, name, label)
        elif vector:
            r = recs_cs(vector, None, label)
        response[label] = r
    return response



def makeup_recommendation(skin_tone, skin_type):
    result = []
    dff = pd.DataFrame()
    dff = pd.concat([dff, makeup[(makeup['skin tone'] == skin_tone) & (makeup['skin type'] == skin_type) & (makeup['label'] == 'foundation')].head(2)])
    dff = pd.concat([dff, makeup[(makeup['skin tone'] == skin_tone) & (makeup['skin type'] == skin_type) & (makeup['label'] == 'concealer')].head(2)])
    dff = pd.concat([dff, makeup[(makeup['skin tone'] == skin_tone) & (makeup['skin type'] == skin_type) & (makeup['label'] == 'primer')].head(2)])
    dff= dff.sample(frac = 1)
    data = dff[['brand', 'name', 'price', 'url', 'img', 'skin type', 'skin tone']].to_dict('split')['data']
    for element in data:
        result.append(wrap_makeup(element))
    return result

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
        pred_class1 = class_names1[tf.argmax(pred1[0])].lower().replace('_skin', '')
    else:
        pred_class1 = class_names1[int(tf.round(pred1[0]))].lower().replace('_skin', '')
    return pred_class1

def prediction_acne(img_path):
    new_image = load_image(img_path)
    pred2 = model2.predict(new_image)
    if len(pred2[0]) > 1:
        pred_class2 = class_names2[tf.argmax(pred2[0])]
    else:
        pred_class2 = class_names2[int(tf.round(pred2[0]))]
    return pred_class2
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.parsers import JSONParser

@csrf_exempt
def skin_metrics(request):
    if request.method == 'POST':
        serializer = ImageUploadSerializer(data=request.FILES)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            img_name = f"{uuid.uuid4()}.jpg"
            img_path = os.path.join(settings.MEDIA_ROOT, img_name)
            with open(img_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            skin_type = prediction_skin(img_path)
            acne_type = prediction_acne(img_path)
            tone = identify_skin_tone(img_path, dataset=skin_tone_dataset)
            os.unlink(img_path) 
            
            feature_vector = [0] * len(features)
            feature_vector[features.index(skin_type)] = 1
            if acne_type == 'Severe':
                feature_vector[features.index('acne')] = 1
            
            skincare_recs = recs_essentials(vector=feature_vector)
            
            return JsonResponse({
                'type': skin_type, 
                'tone': str(tone), 
                'acne': acne_type,
                'skincare_recommendations': skincare_recs,
            }, status=200)
        return JsonResponse(serializer.errors, status=400)