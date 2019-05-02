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

test_num = 1000
X_test, y_test = process_images(test_imgs[0:test_num], 256, 256)
x = np.array(X_test)
y = np.array(y_test)

test_datagen = ImageDataGenerator(rescale=1./255)

model = load_model('model_keras.h5')
TP = 0
TN = 0
FP = 0
FN = 0
correct = 0
incorrect = 0
label = ''
for i, batch in enumerate(test_datagen.flow(x, shuffle=False, batch_size=1)):
    pred = model.predict(batch)
    if pred > 0.5:
        label = 'robot'
    else:
        label = 'empty'

    if y[i] == 1 and label == 'robot':
        correct += 1
        TP += 1
    if y[i] == 0 and label == 'empty':
        correct += 1
        TN += 1
    if y[i] == 1 and label == 'empty':
        incorrect += 1
        FN += 1
    if y[i] == 0 and label == 'robot':
        incorrect += 1
        FP += 1


    # Uncomment plot code below to see the images throughout the testing process
    if i % 100 == 0:
        print('is robot') if y[i] == 1 else print('is empty')
        plt.title(label)
        plt.imshow(batch[0])
        plt.show()
    if i+1 == test_num:
        print('Correct', correct)
        print('Incorrect', incorrect)
        print('Accuracy', str(correct/(correct+incorrect)*100) + '%')
        print('TP:', TP, 'FP:', FP)
        print('FN:', FN, 'TN:', TN)
        exit()