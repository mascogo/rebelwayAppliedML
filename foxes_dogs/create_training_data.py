import os
import pickle
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
import cv2

DATADIR = r'C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\dev\rebelwayAppliedML\foxes_dogs\Animal Image Dataset-Cats, Dogs, and Foxes'
CATEGORIES = ['fox', 'dog']
IMG_SIZE = 50

def create_tr_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # Assign a number to each category
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_array, class_num])
            except Exception as e:
                print(f"Error processing image {img}: {e}")

    random.shuffle(training_data)
    return training_data

training_data = create_tr_data()
print(f"Total training data created: {len(training_data)}") 

X = []
y = []  

for features, label in training_data:
    X.append(features)
    y.append(label) 
    

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for grayscale images

with open('X_pickle.pkl', 'wb') as Xfile:
    pickle.dump(X, Xfile)

with open('y_pickle.pkl', 'wb') as yfile:
    pickle.dump(y, yfile)
