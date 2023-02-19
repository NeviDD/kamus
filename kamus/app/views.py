"""
Definition of views.
"""
import numpy as np
import cv2
import pickle
import cv2
import functools
import numpy as np
from datetime import datetime
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import HttpRequest
from .models import Translate
from .forms import *
from skimage import color
from base64 import b64encode
import matplotlib.pyplot as plt


with open('tulis220.pkl', 'rb') as f:
    az = pickle.load(f)

with open('alphabet135C.pkl', 'rb') as f:
    zz = pickle.load(f)

centroids= az.cluster_centers_


class_mappings = {}
class_mapping = {}                                                                                                                                                    
alphabet = ['P','L','I','C','V','H','Z','J','Y','X','S','W','L','M','J','T','B','N','U','U','O','M',
             'L','V','K','F','W','F','Q','G','E','A','X','Y','T','K','P','C','H','R','Q','Q','R','A','S',
             'D','R','N','F','B','M','D','D','B','M','Q','I','E','N','P','W','A','A','U','D','Z','J','G',                                                    #alphabets = ['S','C','E','A','I','X','V','W','B','H','Z','M','O','N','P','Q','D','U','K','J','R','Y','G','L', 'T','F']#data folder rr,my_dumped_classifer.pkl                                                                                                                                                           #alphabets = ['I','D','G','L','A','N','V','P','Z','W','K','M','S','Q','J','B','X','U','R','T','F','C','Y','H', 'O','E']#data folder characterA,classifer.pkl
             'F','B','U','G','J','Y','K','X','Z','G','X','E','O','H','R','N','O','C','T','C','K','S','G',                                                    #alphabets = ['i','g','d','t','m/y','c/g','e/f','h/n/u','c/o','z','v','b/h','w','n/x','j','u','h','t','a','p','s','g/q','d/j','k','k/r','r','l','m','i/l','w','m','z','b/e/f/l','v/y','e/o','k','a','r','a/d','f/r','q','c/e','n','b/d/u','b','m','q','f','p','p','n','j']#data folder characterA,alphabet52c.pkl
             'T','H','P','I','V','Z','S','K','Y','I','O','A','A','E','T','S','E','Y','X','F','H','D','C',
             'L','R','D','J','B','H','Z','W','V','W','V','O','E','M','C','G','Q','F','I','P','I/L']#data folder characterA,alphabet135c.pkl
alphabets = ['w', 'a/c/o', 'p', 'i', 'h', 'u', 'l', 't', 'r', 'j', 'x', 'o', 'v', 'z', 'c', 'q', 'c', 'v/y', 'o', 'h', 's', 'j', 'z', 'w', 'm', 'q', 'p', 'a', 'l', 't', 'f', 'y', 'k', 'g/s', 'h', 'p', 'e', 'w', 'x', 'l', 'c', 'v', 'n', 'z', 'f', 'z', 'u', 'm', 'c', 'a', 'u', 'n', 'd', 'l', 'f', 'e', 's', 'm', 'n', 'b/h/k', 'd', 'q', 'v', 'h', 'l', 'b', 'g', 'x', 'm', 'x', 'a', 'j', 'm', 'y', 'i', 'f', 'g/j', 'q', 'b', 't', 'e', 'p', 'k', 'e', 'x', 'h', 's', 'q', 'q', 'o', 's', 'm', 'd', 'a', 't', 'b', 't', 'r', 's', 'e', 'k', 'v', 'z', 'a', 'm', 'g', 'a', 'v', 'w', 'k', 'w', 'k', 'q', 'a', 'r', 't', 'd', 'z', 'u', 'g', 'n', 'j', 'n', 'y', 'a', 'v', 'o', 'd/x', 's', 'p', 'z', 'd', 'q', 'h', 'w', 'n', 't', 'q', 'm', 'p/t/w/y', 'v', 'k', 'r', 'l', 'm', 'u', 'p', 'p', 'k', 'f', 'k', 'e', 'h', 's', 'q', 'x', 'e', 'w', 'f', 'c', 'y', 'b', 'u', 'o', 'l', '', 'n', 'i', 'f', 'r', 'g', 'i', '', 'z', 'i', 'g', 'o', 'd', 'b', 'y', 'g', 'd', 'p', 'h', 'f', 'n', 'j', 'w', 'u', 'g', 'f', 'e', 'y', 'b', 't', 'l', 'x', 's', 'j', 'e', 'e', 'k', 'd', 'd', 'o', 'y', 'r', 'r', 'm', 'i', 'j', 'r', 'o', 'c', 'i', 'u', 'c', 's', 'x', '']   #k=220 tulis220.pkl 
#alphabets = ['z', 'a/c/o', 'r', 'l', 'r', 'l', 'b', 'h', 'g', 'j', 'd', 'h', 'l', 'n', 'p', 'f', 'w', 'k', 'v', 'r', 's', 'x', 'm', 'v', 'x', 'c/e/l', 'u', 'j', 'g', 'i', 'h', 'q', 't', 'w', 'e', 'a', 'u', 'y', 'f', 'i', 'm', 'b', 'w', 'y', 'e', 'v', 't', 'n', 'e', 'q', 'e', 'z', 'u', 'z', 'k', 'p', 't', 'a', 'j', 'm', 'd', 'x', 'n', 'd', 'd/m/t/w/y/z', 'z', 't', 'v', 'd', 'g/j/v/y', 'c', 'n', 'c', 's', 'i', 'h', '', 'z', 'e', 'i', 'o', 'q', 'n', 'p', 'p', 'x', 'b', 'f', 'w', 'a', 's', 'b', 'o', 'q', 's', 'f', 'q', 'a', 'v', 'k', 'p', 'n', 's', 'r', 'k', 'j', 'z', 'm', 'c/o', 'e', 'e', 'd', 'q', 'x', 'k', 'a', 'z', 'g', 'x', 'b/s', 'a', 'g', 'k', 'q', 'd', 'w', 'm', 'e', 'a', 'b', 'j', 't', 'x', 'q', 'y', 'a', 'p', 'i/l', 'w', 'r', 'u', 'w', 'm', 'h', 'b', 'f', 'o', 'y', 'n', 'h', 'm', 't', 's', 'e', 'n', 'p', 'q', 'i', 'h', 'd', 'u', 'n', 'i', 'd', 'f/r', 'o', 'd', 'o', 'p', 'k', 'h', 't', 'q', 'h/k', 'g', 'r', 'c', 'f', 'm', 'g', 'u', 'o', 'y', 'p', 'r', 'k', 'o', 'v', 'g', 's', 'h', 'g', 'f', 'c', 't', 'f', 'x', 'l', 'k', 'b']   #k=200 tulisjadi.pkl 
"""alphabets = ['e', 'i/l', 'k', 'd', 'j', 'n', 'u', 'r', 't', 'c', 'q', 'i', 'v', 'r', 'h', 'l', 'r', 'm', 'a', 'x', 'f', 'x', 'j', 'w', 'b', 'z', 't', 'o', 'w', 'v', 'a', 'c', 'c', 'i', 'h', 'e', 's', 'z', 'y', 'm', 'm', 'z', 'x', 'z', 'y', 'h', 'g', 't', 'h', 'w', 'r', 'd', 'u', 's', 'a', 'm', 'p', 'g', 'c', 'f', 'j', 'e', 'l', 'u', 'f', 'v', 'f', 'g', 'c', 'k', 'n', 'p', 'd', 'd', 'x', 'm', 'a', 'a', 'e', 'w', 'd', 'n', 'p/e', 'd', 'q', 'h', 's', 'q', 'm', 'o', 'e', 't', 'b', 'q', 'y', 'q', 'y', 'w', 'a', 'k', 'k', 'l', 'f', 'q', 'r', 'n', 'g', 'a', 'z', 'g', 'b', 'd', 'n', 'm', 'i', 'u', 'e', 'g', 'k', 'z', 't', 'u', 'h', 's', 'o', 'e', 'e', 'v', 'y', 'x', 'j', 'x', 'd', 'l', 'o', 'v', 'b', 'g', 's', 'z', 'o', 'c', 'l', 'l', 'b', 'p', 'w', 'k', 'y', 'k', 'o', 'i', 'q', 'n', 'f', 'y', 'k', 'j', 's', 't', 'c', 'u', 's', 'r', 'n', 'u', 'p', 's', 'r', 'f', 'x', 'q', 'o', 'v', 'i', 'h', 'p', 'b', 'b', 'p', 'x', 'v', 'l', 'j', 'y', 'b', 'u', 'n', 'h', 'w', 'g', 'f', 'r', 'z', 'm', 'a', 't', 'j', 'p', 'o', 'w', 'c'] #k=200 tulistangan.pkl"""
for i in range(len(alphabets)):
    class_mappings[i] = alphabets[i]

for i in range(len(alphabet)):
    class_mapping[i] = alphabet[i]

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


def konversiIndonesia(cari_kata):
    assert isinstance(cari_kata, HttpRequest)
   
   
    cari_query = cari_kata.GET.get('cari_kata', None)
    
    
    if cari_query != '' and cari_query is not None:
        qs = Translate.objects.filter(
            models.Q(Indonesia__icontains=cari_query) |
            models.Q(Sunda_Lemes__icontains=cari_query) |
            models.Q(Sunda_Sedang__icontains=cari_query)
        )
        qss = Translate.objects.filter(
            models.Q(Indonesia__exact=cari_query) |
            models.Q(Sunda_Lemes__exact=cari_query) |
            models.Q(Sunda_Sedang__exact=cari_query)
        )
        # Dapatkan kolom mana kata ditemukan pada setiap objek
        search_results = []
        for obj in qss:
            results = []
            if cari_query in obj.Indonesia:
                results.append('Indonesia')
            if cari_query in obj.Sunda_Lemes:
                results.append('Sunda_Lemes')
            if cari_query in obj.Sunda_Sedang:
                results.append('Sunda_Sedang')

            search_results.append({'obj': obj, 'results': results})
        context = {
            'queryset' : qs,
            'cari_query' : cari_query,
            'search_results': search_results
        }         
        return render(cari_kata, 'app/index.html', context)
    return render(cari_kata, 'app/index.html')

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
            gray = cv2.bilateralFilter(gray, 10, 10, 10)
            
            frame_buff= cv2.imencode('.jpg', gray)[1]
            encode = b64encode(frame_buff)
            decode = encode.decode('ascii')   
            mime = "image/jpg"
            uriii = "data:%s;base64,%s" % (mime, decode)

            # Apply  thresholding 
            (thresh, threshh) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            #cv2.namedWindow('haha', cv2.WINDOW_NORMAL)
            #cv2.imshow('haha', thresh)
            #cv2.waitKey(0)
            thresh = 255-threshh
            #thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
            frame_buff= cv2.imencode('.jpg', thresh)[1]
            encode = b64encode(frame_buff)
            decode = encode.decode('ascii')   
            mime = "image/jpg"
            uri = "data:%s;base64,%s" % (mime, decode)
            plt.imshow(thresh)
            plt.show()
            #erode 
            #Some noise reduction
            #img_erode = cv2.erode(thresh, np.ones((1,2), np.uint8))
            # Find contours and get bounding box for each contour
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            # Sort the bounding boxes from left to right, top to bottom
            # sort by Y first, and then sort by X if Ys are similar
            def compare(rect1, rect2):
                if abs(rect1[0] - rect2[0]) <= 15:       
                    return rect1[1] - rect2[1]
                else:
                    return rect1[0] - rect2[0]
            boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

            TARGET_WIDTH = 28
            TARGET_HEIGHT = 28    
           
            action = request.POST.get('action')
            if action == 'action1':
                
                huruf = ""
                # Loop over the bounding boxes
                for rect in boundingBoxes:
                    # Get the coordinates from the bounding box
                    x,y,w,h = rect
                    # Crop the character from the mask
                    crop = image[y:y+h, x:x+w]
                    #cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
                    #cv2.imshow('thresh',crop)
                    #cv2.waitKey(0)
                    #Inverting the image
                    # the characters are black on a white background
                    #crop = cv2.bitwise_not(crop)   
                
                
                    # Apply padding 
                    crop = cv2.copyMakeBorder(crop,0,0,0,0, cv2.BORDER_CONSTANT, None, value=[255, 255, 255])
                    #cv2.namedWindow('padding', cv2.WINDOW_NORMAL)
                    #cv2.imshow('padding', crop)
                    #cv2.waitKey(0)
                    # Convert and resize image
                    #crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
                    crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
                    crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                    #crop= color.rgb2gray(crop)
                    #cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
                    #cv2.imshow('gray', crop)
                    #cv2.waitKey(0)
                    (thresh, crop) = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
                    #crop=255-crop
                    #cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
                    #cv2.imshow('Final', crop)
                    #cv2.waitKey(0)
                    #cv2.imshow('Final', crop)
                    #cv2.waitKey(0)
                    # Prepare data for prediction
                
                
                    crop = crop.reshape(1, 28*28)
                    crop = crop.astype("float32") / 255.0
                    print(type(crop))
                    #prediction
                    predicted_cluster = az.predict(crop)
                    #mapping int to a-z
                    predicted_cluster = class_mappings[int(predicted_cluster)]                                                                                                                                  
                 
                    huruf += predicted_cluster
                    kotak = cv2.rectangle(threshh, (x,y), (x+w,y+h), (0, 255, 0), 2)
                    frame_buff= cv2.imencode('.jpg', kotak)[1]
                    encode = b64encode(frame_buff)
                    decode = encode.decode('ascii')
                    mime = "image/jpg"
                    urii = "data:%s;base64,%s" % (mime, decode)
                    qs = Translate.objects.filter(
                    models.Q(Indonesia__icontains=huruf) |
                    models.Q(Sunda_Lemes__icontains=huruf) |
                    models.Q(Sunda_Sedang__icontains=huruf)
                    )
                    qss = Translate.objects.filter(
                        models.Q(Indonesia__exact=huruf) |
                        models.Q(Sunda_Lemes__exact=huruf) |
                        models.Q(Sunda_Sedang__exact=huruf)
                    )
                    # Dapatkan kolom mana kata ditemukan pada setiap objek
                    search_results = []
                    for obj in qss:
                        results = []
                        if huruf in obj.Indonesia:
                            results.append('Indonesia')
                        if huruf in obj.Sunda_Lemes:
                            results.append('Sunda_Lemes')
                        if huruf in obj.Sunda_Sedang:
                            results.append('Sunda_Sedang')

                        search_results.append({'obj': obj, 'results': results})
                        print('Result:berhasil', huruf)
                return render(request, 'app/fiturocr.html', {'text': huruf, 'qs': qs,'input_image' : input_image, 
                                                            'uri' : uri, 'urii' : urii, 'uriii' : uriii, 'queryset' : qs, 'search_results': search_results} )
            
            elif action == 'action2':
                
                huruf = ""
                # Loop over the bounding boxes
                for rect in boundingBoxes:
                    # Get the coordinates from the bounding box
                    x,y,w,h = rect
                    # Crop the character from the mask
                    crop = image[y:y+h, x:x+w]
                    #cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
                    #cv2.imshow('thresh',crop)
                    #cv2.waitKey(0)
                    #Inverting the image
                    # the characters are black on a white background
                    #crop = cv2.bitwise_not(crop)   
                    # Apply padding 
                    crop = cv2.copyMakeBorder(crop,1,1,1,1, cv2.BORDER_CONSTANT, None,255)
                    #cv2.namedWindow('padding', cv2.WINDOW_NORMAL)
                    #cv2.imshow('padding', crop)
                    #cv2.waitKey(0)
                    # Convert and resize image
                    #crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
                    crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
                    crop= color.rgb2gray(crop) 
                    #crop= color.rgb2gray(crop)
                    #cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
                    #cv2.imshow('gray', crop)
                    #cv2.waitKey(0)                   
                    #crop=255-crop
                    #cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
                    #cv2.imshow('Final', crop)
                    #cv2.waitKey(0)
                    #cv2.imshow('Final', crop)
                    #cv2.waitKey(0)
                    # Prepare data for prediction
                
                
                    crop = crop.reshape(1, 28*28)
                    crop = crop.astype("float32") / 255.0
                    print(type(crop))
                    #prediction
                    predicted_cluster = zz.predict(crop)
                    #mapping int to a-z
                    predicted_cluster = class_mapping[int(predicted_cluster)]                                                                                                                                  
                 
                    huruf += predicted_cluster
                    kotak = cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)
                    frame_buff= cv2.imencode('.jpg', kotak)[1]
                    encode = b64encode(frame_buff)
                    decode = encode.decode('ascii')
                    mime = "image/jpg"
                    urii = "data:%s;base64,%s" % (mime, decode)
                    qs = Translate.objects.filter(
                    models.Q(Indonesia__icontains=huruf) |
                    models.Q(Sunda_Lemes__icontains=huruf) |
                    models.Q(Sunda_Sedang__icontains=huruf)
                    )
                    qss = Translate.objects.filter(
                        models.Q(Indonesia__exact=huruf) |
                        models.Q(Sunda_Lemes__exact=huruf) |
                        models.Q(Sunda_Sedang__exact=huruf)
                    )
                    # Dapatkan kolom mana kata ditemukan pada setiap objek
                    search_results = []
                    for obj in qss:
                        results = []
                        if huruf in obj.Indonesia:
                            results.append('Indonesia')
                        if huruf in obj.Sunda_Lemes:
                            results.append('Sunda_Lemes')
                        if huruf in obj.Sunda_Sedang:
                            results.append('Sunda_Sedang')

                        search_results.append({'obj': obj, 'results': results})
                        print('Result:berhasil', huruf)
                return render(request, 'app/fiturocr.html', {'text': huruf, 'qs': qs,'input_image' : input_image, 
                                                            'uri' : uri, 'urii' : urii, 'uriii' : uriii, 'queryset' : qs, 'search_results': search_results} )
            # do action 2 with the image
                
           
            #qs = Translate.objects.all()
        
            
            #qs = qs.filter(Indonesia=huruf)
      
    else:
        form = InputImageForm()
    return render(request, 'app/fiturocr.html', {'form' : form})
    
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
