import os,cv2
path = r'C:\Users\diven\Desktop\TulisanTangan\A' # Source Folder
dstpath = r'C:\Users\diven\Desktop\graybaru' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")

# Folder won't used
files = os.listdir(path)
from skimage import color

for image in files:
    img = cv2.imread(os.path.join(path,image))
    img = cv2.resize(img, (28, 28)) #resize
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(dstpath,image),blackAndWhiteImage)

    #cv2.imshow('Gray image', gray)
    #cv2.imshow('bw', blackAndWhiteImage)
    #cv2.waitKey(0)
