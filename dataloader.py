from config import *
import glob
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class dataload(keras.utils.Sequence):
    def __init__(self,paths,batch_size,image_shape,):
        super().__init__()

        self.paths = paths
        self.batch_size = batch_size
        self.image_shape = image_shape
        
    def __len__(self):
        return math.ceil(len(self.paths)/self.batch_size)
    

    def __getitem__(self, key):
        batch_idx = self.paths[self.batch_size*key:self.batch_size*(key+1)] # batch_size*current index to batch_size*current index + 1

        x = []
        y = []
        for  path in batch_idx:
            #print(path)
            img = keras.preprocessing.image.load_img(path, target_size=self.image_shape)
            img = img.resize(self.image_shape)
            x.append(np.array(img))
            name = path.split("/")[-1].split("\\")[-2]
            if name == "Cat":
            
                y.append(1)
            else:
                y.append(0)
        y=to_categorical(y, num_classes = 2)

        
        return tf.convert_to_tensor(x), y

cat_dir = glob.glob("C:/Users/Amzad/Desktop/keras_project/image_classification/data/train/Cat/*.*")
dog_dir = glob.glob("C:/Users/Amzad/Desktop/keras_project/image_classification/data/train/Dog/*.*")
train_dir = np.array(cat_dir + dog_dir)
np.random.shuffle(train_dir)

v_cat_dir = glob.glob("C:/Users/Amzad/Desktop/keras_project/image_classification/data/train/Cat/*.*")
v_dog_dir = glob.glob("C:/Users/Amzad/Desktop/keras_project/image_classification/data/train/Dog/*.*")
val_dir = np.array(v_cat_dir + v_dog_dir)
np.random.shuffle(val_dir)



train_ds = dataload(train_dir, batch_size=batch_size, image_shape=(height, width))
val_ds = dataload(train_dir, batch_size=batch_size, image_shape=(height, width))


# train_ds = keras.utils.image_dataset_from_directory(
#     directory='{}/train'.format(data_dir),
#     labels='inferred',
#     label_mode='categorical',
#     batch_size=batch_size,
#     image_size=(height, width))

# val_ds= keras.utils.image_dataset_from_directory(
#     directory='{}/val'.format(data_dir),
#     labels='inferred',
#     label_mode='categorical',
#     batch_size=batch_size,
#     image_size=(height, width))