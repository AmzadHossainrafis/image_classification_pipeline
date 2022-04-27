import glob
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matrice import recall
from config import *
from dataloader import dataload
#from keras.utils import load_model


test_cat_dir = glob.glob("C:/Users/Amzad/Desktop/keras_project/image_classification/data/test/Cat/*.*")
v_dog_dir = glob.glob("C:/Users/Amzad/Desktop/keras_project/image_classification/data/train/dog/*.*")
test_dir = np.array(test_cat_dir + v_dog_dir)

test_ds = dataload(test_dir, batch_size=batch_size, image_shape=(height, width))


model=keras.models.load_model(load_model_path,custom_objects={'recall':recall})
model.compile(
    metrics=["accuracy",keras.metrics.Precision()],
)


model.evaluate(test_ds)


for i in test_ds:
    print(i[0])
    print(model.predict(i[0]))
    break
