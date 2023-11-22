# Get the Feature vector using the pre-trained model VGG16 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

model_vgg16 = VGG16(weights='imagenet', include_top=False)

for layer in model_vgg16.layers:
    layer.trainable = False
    
def ExtractImageFeature(img):
    image = load_img(img, target_size=(224, 224))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_vgg16.predict(x)
    flat_features = features.flatten()
    return flat_features
"""
#dataset 1
folder_path = 'C:\\Users\\haider\\Desktop\\grad_poject\\data\\dataset1\\Training'
folders_list = os.listdir(folder_path)

images_folders_path = []
for folder in folders_list:
    images_folder = os.path.join(folder_path,folder)
    images_folders_path.append(images_folder)  


images_path1 = []
for path in images_folders_path:
    images = os.listdir(path)
    for img in images:
        images_path1.append(os.path.join(path,img))
"""

#dataset 2
folder_path = 'C:\\Users\\haider\\Desktop\\grad_poject\\data\\dataset2\\sample\\images'
folders_list = os.listdir(folder_path)

images_path2 = []
for img in folders_list:
    image_path = os.path.join(folder_path,img)
    images_path2.append(image_path)  



feature_lists = [ExtractImageFeature(img) for img in images_path2]
feature_vector = np.vstack(feature_lists)

np.save('code\\final1-VGG16\\feature_vector2.npy',feature_vector)

"""for i in range(5):
    image = cv2.imread(images_path[i])
    plt.subplot(1,5,i+1)
    plt.imshow(image)
    print(image)
plt.show()"""