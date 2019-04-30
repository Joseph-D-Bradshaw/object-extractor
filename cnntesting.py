import cv2
import numpy as np
# Graphical plotting imports
import matplotlib.pyplot as plt
# pathing, choice and garbage collection
import os
# machine learning!
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from cnntraining import process_images
from random import shuffle

test_dir = './dataset/testing'
test_imgs = ['./dataset/testing/{}'.format(i) for i in os.listdir(test_dir)]
shuffle(test_imgs)

test_num = 10
X_test, y_test = process_images(test_imgs[0:test_num], 256, 256) #y_test will be empty as 
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)

model = load_model('model_keras.h5')
label = ''
for i, batch in enumerate(test_datagen.flow(x, batch_size=1)):
    pred = model.predict(batch)
    if pred > 0.5:
        label = 'robot'
    else:
        label = 'empty'
    plt.title(label)
    plt.imshow(batch[0])
    plt.show()
    if i+1 == test_num:
        break