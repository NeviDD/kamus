from PIL import Image
import numpy as np
import sys
import os
import csv
# default format can be changed as needed
def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    labels = []
    names = []
    keywords = {"a" : "0","b": "1","c" : "2","d": "3","e" : "4","f": "5","g" : "6","h": "7","i" : "8","j": "9","k" : "10","l": "11","m" : "12","n": "13","o" : "14","p": "15","q" : "16","r": "17","s" : "18","t": "19","u" : "20","v": "21","w" : "22","x": "23","y" : "24","z": "25",} # keys and values to be changed as needed
for root, dirs, files in os.walk(myDir, topdown=True):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
            for keyword in keywords:
                if keyword in name:
                    labels.append(keywords[keyword])
                else:
                    continue
            names.append(name)
    return fileList, labels, names
# load the original image
myFileList, labels, names  = createFileList('/content/')
i = 0
for file in myFileList:
    print(file)
    img_file = Image.open(file)
    # img_file.show()
# get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode
# Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()
# Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((width, height))
    value = value.flatten()
    
    value = np.append(value,labels[i])
    i +=1
    
    print(value)
    with open("alfabet.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)

