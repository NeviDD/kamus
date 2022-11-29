"""
Definition of views.
"""
import os 
import keras
import numpy as np
import tensorflow as tf
import cv2
# Data 
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator 
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import seaborn as sns
# Data Visualization 
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Model 
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, GlobalAvgPool2D

# Callbacks 
from keras.callbacks import EarlyStopping, ModelCheckpoint

from skimage import color
from skimage import io
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pickle

import cv2
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import functools
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import HttpRequest
from .models import Translate
from .forms import *
from PIL import Image

with open('alphabet135C.pkl', 'rb') as f:
    az = pickle.load(f)

centroids= az.cluster_centers_

print(az.labels_)
class_mapping = {}
#alphabets = ['S','C','E','A','I','X','V','W','B','H','Z','M','O','N','P','Q','D','U','K','J','R','Y','G','L', 'T','F']#data folder rr,my_dumped_classifer.pkl
#alphabets = ['I','D','G','L','A','N','V','P','Z','W','K','M','S','Q','J','B','X','U','R','T','F','C','Y','H', 'O','E']#data folder characterA,classifer.pkl
alphabets = ['P','L','I','C','V','H','Z','J','Y','X','S','W','I/L','M','J','T','B','N','U','U','O','M','L','V','K','F','W','F','Q','G','E','A','X','Y','T','K','P','C','H','R','Q','Q','R','A','S','D','R','N','F','B','M','D','D','B','M','Q','I','E','N','P','W','A','A','U','D','Z','J','G','F','B','U','G','J','Y','K','X','Z','G','X','E','O','H','R','N','O','C','T','C','K','S','G','T','H','P','I','V','Z','S','K','Y','I','O','A','A','E','T','S','E','Y','X','F','H','D','C','I/L','R','D','J','B','H','Z','W','V','W','V','O','E','M','C','G','Q','F','I/L','P','I/L']#data folder characterA,alphabet135c.pkl
#alphabets = ['i','g','d','t','m/y','c/g','e/f','h/n/u','c/o','z','v','b/h','w','n/x','j','u','h','t','a','p','s','g/q','d/j','k','k/r','r','l','m','i/l','w','m','z','b/e/f/l','v/y','e/o','k','a','r','a/d','f/r','q','c/e','n','b/d/u','b','m','q','f','p','p','n','j']#data folder characterA,alphabet52c.pkl


for i in range(len(alphabets)):
    class_mapping[i] = alphabets[i]


def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    translate = Translate.objects.all()
    context = {
        'queryset' : translate
     }
    return render(
        request,
        'app/index.html', context,       
    )


def konversiIndonesia(textindonesia):
    assert isinstance(textindonesia, HttpRequest)
   
    qs = Translate.objects.all()
    cari_query = textindonesia.GET.get('textindonesia', None)
    
    
    if cari_query != '' and cari_query is not None:
        qs = qs.filter(Indonesia=cari_query)
    context = {
        'queryset' : qs
        }
          
    return render(textindonesia, 'app/index.html', context)

def konversiSundaLemes(textsundalemes):
    assert isinstance(textsundalemes, HttpRequest)
   
    qs = Translate.objects.all()
    cari_query = textsundalemes.GET.get('textsundalemes', None)
    
    
    if cari_query != '' and cari_query is not None:
        qs = qs.filter(Sunda_Lemes=cari_query)
    context = {
        'queryset' : qs
        }
          
    return render(textsundalemes, 'app/index.html', context)

def konversiSundaSedang(textsundasedang):
    assert isinstance(textsundasedang, HttpRequest)
   
    qs = Translate.objects.all()
    cari_query = textsundasedang.GET.get('textsundasedang', None)
    
    
    if cari_query != '' and cari_query is not None:
        qs = qs.filter(Sunda_Sedang=cari_query)
    context = {
        'queryset' : qs
        }
          
    return render(textsundasedang, 'app/index.html', context)

def process_image(request):  
    if request.method == 'POST':
        form = InputImageForm( request.POST, request.FILES)
  
        if form.is_valid():
            from base64 import b64encode
            file = request.FILES['image']
            response = file.read()
            #for display it to website
            encoded = b64encode(response).decode('ascii')
            mime = "image/jpg"
            mime = mime + ";" if mime else ";"
            input_image = "data:%sbase64,%s" % (mime, encoded)
            #numpay array so it can be process
            image = cv2.imdecode(np.fromstring(response, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blurring and thresholding 

            thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
            frame_buff= cv2.imencode('.jpg', thresh)[1]
            encode = b64encode(frame_buff)
            decode = encode.decode('ascii')
            
            mime = "image/jpg"
            uri = "data:%s;base64,%s" % (mime, decode)
            #plt.imshow(thresh)
            #plt.show()
            #erode 
            img_erode = cv2.erode(thresh, np.ones((1,2), np.uint8))
            # Find contours and get bounding box for each contour
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            # Sort the bounding boxes from left to right, top to bottom
            # sort by Y first, and then sort by X if Ys are similar
            def compare(rect1, rect2):
                if abs(rect1[1] - rect2[1]) <= 15:
                    return rect1[0] - rect2[0]
                else:
                    return rect1[1] - rect2[1]
            boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

            TARGET_WIDTH = 28
            TARGET_HEIGHT = 28

            huruf = ""
            # Loop over the bounding boxes
            for rect in boundingBoxes:
                # Get the coordinates from the bounding box
                x,y,w,h = rect
                # Crop the character from the mask
                # and apply bitwise_not because in our training data for pre-trained model
                # the characters are black on a white background
                crop = thresh[y:y+h, x:x+w]
                crop = cv2.bitwise_not(crop)
                # Apply padding 
                crop = cv2.copyMakeBorder(crop, 1,1,1,1, cv2.BORDER_CONSTANT, None, 255)
   
                # Convert and resize image
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)     
                crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
                cv2.imshow('Final', crop)
                cv2.waitKey(0)
                # Prepare data for prediction
                from skimage import color
                crop= color.rgb2gray(crop)
    
                crop = crop.reshape(1, 28*28)
                crop = crop.astype("float32") / 255.0

                #prediction
                #prediction

                predicted_cluster = az.predict(crop)
                #mapping int to a-z
                predicted_cluster = class_mapping[int(predicted_cluster)]

                print(predicted_cluster)
    
    
                huruf += predicted_cluster
                kotak = cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)
                frame_buff= cv2.imencode('.jpg', kotak)[1]
                encode = b64encode(frame_buff)
                decode = encode.decode('ascii')
                mime = "image/jpg"
                urii = "data:%s;base64,%s" % (mime, decode)
            

            from django.db.models import F
            qs = Translate.objects.all()

            
            qs = qs.filter(Indonesia=huruf)
            
            
            print('Result:berhasil', huruf)
            return render(request, 'app/fiturocr.html', {'text': huruf, 'qs': qs,'input_image' : input_image, 'uri' : uri, 'urii' : urii } )
    else:
        form = InputImageForm()
    return render(request, 'app/fiturocr.html', {'form' : form})
    
def success(request):
    return HttpResponse('successfully uploaded')

def grab_image(stream=None):
    data=stream.read()


def bantuan(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/bantuan.html',
        {
            'title':'Bantuan',
            'message':'Your contact page.',
            'year':datetime.now().year,
        }
    )

def fiturocr(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/fiturocr.html',
        {
            'title':'FiturOCR',
            'message':'Your application description page.',
            'year':datetime.now().year,
        }
    )
