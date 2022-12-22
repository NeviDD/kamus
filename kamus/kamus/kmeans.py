# Common times new roman data
import os 
import numpy as np
import cv2
# Data 
import seaborn as sns
# Data Visualization 
import matplotlib.pyplot as plt

from skimage import color
from skimage import io
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import functools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans

img_folder= 'CHARACTERA'

def create_dataset(img_folder):
   
    data=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)        
            data.append(image)
            class_name.append(dir1)
    return data, class_name

# extract the image array and class name
img_data, class_name =create_dataset('CHARACTERA')


target_dict={k: v for v, k in enumerate(np.unique(class_name))}
print(target_dict)

target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
print(target_val)



#plt.gray() # B/W images

plt.figure(figsize = (15,8)) # Adjusting figure size

# Displaying a grid of 3x3 images

""" showing dataset randomly
for i in range(1,11):
  id = np.random.randint(len(img_data))
  image, label = tf.squeeze(img_data[id]), class_mapping[int(target_val[id])]
  plt.subplot(2,5,i)
  plt.imshow(image, cmap='binary')
  plt.title(label)
  plt.axis('off')

plt.tight_layout()   
plt.show()
"""


img_data=np.array(img_data)
target_val=np.array(target_val)



#data normalization
img_data=img_data/255.0
#reshape data
img_data = img_data.reshape(len(img_data),-1)

#kmeans
n_clusters = 120
kmeans =MiniBatchKMeans(n_clusters)
az=kmeans.fit(img_data)
Z=kmeans.predict(img_data)
#print(kmeans.cluster_centers_.shape)
import pickle
#print(az)
#save the classifier
with open('120.pkl', 'wb') as fid:
   pickle.dump(az, fid)


"""
#mengecek isi cluster
for i in range(0,n_clusters):

    row = np.where(Z==i)[0]       # row in Z for elements of cluster i
    num = row.shape[0]            #  number of elements for each cluster
    r = int(np.floor(num/10.))    # number of rows in the figure of the cluster 

    print("cluster "+str(i))
    print(str(num)+" elements")

    plt.figure(figsize=(10,10))
    for k in range(0, num):
        plt.subplot(r+1, 10, k+1)
        image = img_data[row[k], ]
        image = image.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()
"""

def retrieve_info(cluster_labels,target_val):
  # Initializing 
  reference_labels = {}
  # For loop to run through each label of cluster label
  for i in np.unique(kmeans.labels_):
   index = np.where(cluster_labels == i,1,0)
   num = np.bincount(target_val[index==1]).argmax()
   reference_labels[i] = num
  return reference_labels

reference_labels = retrieve_info(kmeans.labels_,target_val)

print(reference_labels)

alphabet_labels = np.random.rand(len(kmeans.labels_))

for i in range(len(kmeans.labels_)):

  alphabet_labels[i] = reference_labels[kmeans.labels_[i]]



# Accuracy Score for right predictions
print(accuracy_score(alphabet_labels,target_val))

""" cross validation tidak bisa
from sklearn.model_selection import cross_val_score
scores = cross_val_score(az, img_data, target_val, cv=5)
print(scores)
"""

#matrix confusion
mat = confusion_matrix(target_val,alphabet_labels)
sns.heatmap(mat, annot=True)
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.show()


# elbow method
distortions = []
n_clusters = range(52,150)
for k in n_clusters:
    kmeans =MiniBatchKMeans(n_clusters=k)
    az=kmeans.fit(img_data)
    Z=kmeans.predict(img_data)
    distortions.append(kmeans.inertia_)                                                
   #print(kmeans.cluster_centers_.shape)                                                           
plt.figure(figsize=(16,8))
plt.plot(n_clusters, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()








