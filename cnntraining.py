# Keras VGG-16 Model.
# Network architecture: VGGnet

# Data sorting imports
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
# Graphical plotting imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
# pathing, choice and garbage collection
import os
import random
import gc
# machine learning!
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

def process_images(list_of_images, nrows, ncolumns):
    images = []
    labels = []
    for image in list_of_images:
        im = cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(im)
        if 'robot' in image:
            labels.append(1)
        elif 'empty' in image:
            labels.append(0)
    return images, labels

if __name__ == '__main__':
    train_dir = './dataset/training'
    test_dir = './dataset/testing'

    train_robots = ['./dataset/training/{}'.format(i) for i in os.listdir(train_dir) if 'robot' in i]
    train_emptys = ['./dataset/training/{}'.format(i) for i in os.listdir(train_dir) if 'empty' in i]

    test_imgs = ['./dataset/testing/{}'.format(i) for i in os.listdir(test_dir)]

    train_imgs = train_robots[:1500] + train_emptys[:1500]
    random.shuffle(train_imgs)

    del train_robots
    del train_emptys
    gc.collect()

    nrows = 256
    ncolumns = 256
    channels = 3



    print('Processing training images')
    X, y = process_images(train_imgs, nrows, ncolumns)
    del train_imgs

    print(y[:5])

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

    sns.countplot(y)
    plt.title('Labels for Robots and Non Robots (empty)')
    plt.show()

    print('Shape of train images is:', X.shape)
    print('Shape of labels is:', y.shape)
    print('Shape of X_train is', X_train.shape)
    print('Shape of X_val is', X_val.shape)
    print('Shape of y_train is', y_train.shape)
    print('Shape of y_val is', y_val.shape)

    del X
    del y
    gc.collect()

    n_train = len(X_train)
    n_val = len(X_val)

    batch_size = 32

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()
    # RMSprop optimizer with learning rate of 0.0001
    # Binary cross entropy loss function as it's a binary classification (robot or not)
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001), metrics=['acc'])


    # Creation of image data generator to take our synthentic data and further augment it
    # Should help prevent overfitting due to our small dataset size
    # rescale=1./255 normalises image pixel values to zero mean and standard dev of 1, helps model learn.
    train_datagen = ImageDataGenerator(rescale=1./255, #scale image between 0 - 1
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1./255) # Only rescale validation data, no augmentation

    # Create generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    # Training for 64 epochs with n_train/batchsize steps
    # Fits the model based on data being given batch by batch via a Python generator
    history = model.fit_generator(train_generator,
                                    steps_per_epoch=n_train//batch_size,
                                    epochs=64,
                                    validation_data=val_generator,
                                    validation_steps=n_val//batch_size)

    # 64 full passes through data, model will make gradient updates every n_train/batch_size steps
    # Save the trained model
    model.save_weights('model_weights.h5')
    model.save('model_keras.h5')

    # Plot accuracys of training validation and loss
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()

    # Now this code is not a "good test" as it is tested on synthetic data
    X_test, y_test = process_images(test_imgs[0:10]) #y_test will be empty as 
    x = np.array(X_test)
    test_datagen = ImageDataGenerator(rescale=1./255)

    i = 0
    text_labels = []
    plt.figure(figsize=(30,20))
    for batch in test_datagen.flow(x, batch_size=1):
        pred = model.predict(batch)
        if pred > 0.5:
            text_labels.append('robot')
        else:
            text_labels.append('empty')
        plt.subplot(5 / ncolumns + 1, ncolumns, i + 1)
        plt.title('This is a ' + text_labels[i])
        img_plot = plt.imshow(batch[0])
        i += 1
        if i == 10:
            break
    plt.show()
