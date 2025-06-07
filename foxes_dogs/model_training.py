from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import pickle
import numpy # as np
import time
import  os


pX_path = r'C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\dev\rebelwayAppliedML\foxes_dogs\X_pickle.pkl'
py_path = r'C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\dev\rebelwayAppliedML\foxes_dogs\y_pickle.pkl'


pX = pickle.load(open(pX_path, 'rb'))
py = pickle.load(open(py_path, 'rb'))
print("Loaded data shapes:")
print("X shape: ", len(pX), "y shape: ", len(py))   

X = numpy.array(pX)
y = numpy.array(py)

# Normalize the data
X = X / 255.0 # Scale pixel values to [0, 1]


dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"Foxes_Dogs-CNN-{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"
            print(NAME)

            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))     
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Flatten())  # Flatten the output of the convolutional layers

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))  # Output layer for binary classification
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            

            model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])
            
             