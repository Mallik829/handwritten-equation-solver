# Import libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist, cifar10


# Load in model here
filename = '../models/cnn_3layer64dropout.pkl'
model = pickle.load(open(filename, 'rb'))

# set equation directory path
eq_dir = '../image_data/image_output/'


# create test data
def create_testing_data(eq_dir):
    img_data = []
    for file in sorted(os.listdir(eq_dir)):
        img_path = os.path.join(eq_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50), interpolation = cv2.INTER_AREA)
        img = np.array(img)
        img = img.astype('float32')
        img_data.append(img)
    X_test = np.array(img_data, np.float32)
    X_test = X_test.reshape(-1, 50, 50, 1)
    return X_test


X_test = create_testing_data(eq_dir)

# get predicted characters
y_pred = np.argmax(model.predict(X_test), axis=1)

# convert prediction output to symbols
math_dict = {0: '+', 1: '-', 2: '0', 3: '1', 4: '2', 5: '3', 6: '4', 7: '5',
             8: '6', 9: '7', 10: '8', 11: '9', 12: '*'}


# create equation
equation = ''
for i in y_pred:
    equation += math_dict[i]
print(equation)

# Solve equation
print(f'{equation} = {eval(equation)}')