
import cv2

import cv2
import functools
import numpy as np
from datetime import datetime
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import HttpRequest
import string
from base64 import b64encode
import matplotlib.pyplot as plt
       
letters = list(string.ascii_lowercase)
# Load the image
image = cv2.imread('z.png')
          
filtered = cv2.bilateralFilter(image, 9, 75, 75)            
gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
          

# Apply  thresholding 
(thresh, thresh) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
thresh = 255-thresh
#thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
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

huruf = ""
# Loop over the bounding boxes
for i, rect in enumerate(boundingBoxes):
    # Get the coordinates from the bounding box
    x,y,w,h = rect
    # Crop the character from the mask
    crop = image[y:y+h, x:x+w]
 
    #Inverting the image
    # the characters are black on a white background
    #crop = cv2.bitwise_not(crop)   
                
                
    # Apply padding 
 
    # Convert and resize image
    #crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
    crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    #crop= color.rgb2gray(crop)
    (thresh, crop) = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
    #crop=255-crop
    cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
    cv2.imshow('Final', crop)
    cv2.waitKey(0)
    #cv2.imshow('Final', crop)
    #cv2.waitKey(0)
    # Prepare data for prediction
    filename= letters[i] + '.png'
    cv2.imwrite(filename,crop)            
                
  