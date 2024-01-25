import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# use any other libraries as needed

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Dense,Flatten,Conv2D,MaxPool1D
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.framework import ops
tf.random.set_seed(1234) 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import math
import numpy as np
#import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
#import scipy
from PIL import Image
import pandas as pd
import splitfolders
import os
tf.get_logger().setLevel('INFO')
boolean_activate_resnet50_home= True


dirname= os.getcwd()
import splitfolders
#splitfolders.ratio("upDatasets", output="Datasets",
#    seed=1337, ratio=(.85, .1, .05), group_prefix=None, move=False)
print('Split Done.')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE+(3,)

directory_train = "Datasets/train"
directory_val = "Datasets/val"
directory_test = "Datasets/test"
train_dataset = image_dataset_from_directory(directory_train,
                                             label_mode='categorical',
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)

validation_dataset = image_dataset_from_directory(directory_val,
                                             label_mode='categorical',
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)

test_dataset = image_dataset_from_directory(directory_test,
                                             label_mode='categorical',
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             seed=42)

class_name = train_dataset.class_names
print(class_name)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def data_augmenter():
    
    l1 = tf.keras.layers.RandomFlip(mode='horizontal')
    l2 = tf.keras.layers.RandomRotation(0.2)
    l3 = tf.keras.layers.RandomZoom(0.2)
    l4 = tf.keras.layers.RandomContrast(0.2)
    l5 = tf.keras.layers.RandomBrightness(0.2)
    with tf.device('/cpu:0'):
        data_augmentation = tf.keras.Sequential([])
        data_augmentation.add(l1)
        data_augmentation.add(l2)
        data_augmentation.add(l3)
        data_augmentation.add(l4)
        data_augmentation.add(l5)
    
    return data_augmentation

data_augmentation = data_augmenter()

def tf_model(img_shape):
    #RGB images, so 3 channels
    input_shape = img_shape + (3,)
    
    #Import mobile net
    #base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    #base_model = tf.keras.applications.MobileNet(input_shape=input_shape, weights='imagenet',input_tensor=None)
    #base_model = tf.keras.applications.mobilenet_v3.MobileNetV3(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model = tf.keras.applications.EfficientNetB4(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    #Create inputs
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = tfl.Normalization()(inputs)

    x = base_model(x, training=False)
    x = tfl.GlobalAveragePooling2D()(x) 
    x = tfl.Dropout(rate=0.2)(x)

    outputs = tfl.Dense(len(class_name), activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

model1=tf_model(IMG_SIZE)
model1.summary()


base_learning_rate = 0.01
rate_of_deceleration = 10 #larger number -> the output get smaller slower
func = lambda epoch: (1/((epoch/rate_of_deceleration)+1))*base_learning_rate

lr_schedule = tf.keras.callbacks.LearningRateScheduler(func)


model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 20
history = model1.fit(train_dataset, validation_data=validation_dataset, epochs=20)

model1.evaluate(test_dataset)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model1)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the model.
with open('apml.tflite', 'wb') as f:
  f.write(tflite_quant_model)


