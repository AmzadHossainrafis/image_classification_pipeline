from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.densenet import DenseNet121
import tensorflow as tf
from tensorflow.keras import layers
from config import *

def model_chg(model_name=model_name,transfer_learing=True,height=height,width=width,channels=channels,classes=num_classes):
    if model_name=="inception_v3":
        print(f'model_name :{model_name}')
        print(f"number of epochs :{epochs}")
        print(f"batch size :{batch_size}")
        print(f"image height :{height} width :{width} channels :{channels}")
        if transfer_learing:
            print("transfer_learning : True ")
            main_model=InceptionV3(input_shape=(height,width,channels))
            main_model.layers.pop()
            model=tf.keras.Sequential()
            for layer in main_model.layers:
                model.add(layer)
            for layer in main_model.layers:
                layer.trainable=False
            model.add(layers.Dense(num_classes,activation="softmax"))
        
        else:
            model = InceptionV3(weights=None,input_shape=(height,width,channels),classes=classes)  
        return model

    elif model_name=="resnet50": 
        model = ResNet50(weights=None,input_shape=(height,width,channels),classes=classes)  
        return model





    elif model_name=="vgg16":

        print(f'model_name :{model_name}')
        print(f"number of epochs :{epochs}")
        print(f"batch size :{batch_size}")
        print(f"image height :{height} width :{width} channels :{channels}")
        if transfer_learing: 
            main_model=VGG16(input_shape=(height,width,channels))
            main_model.layers.pop()
            model = tf.keras.Sequential()
            for layer in main_model.layers:
                model.add(layer)
            for layer in main_model.layers:
                layer.trainable=False
            model.add(layers.Dense(num_classes, activation='softmax'))
        else:
            model = VGG16(weights=None ,input_shape=(height,width,channels),classes=classes)
        return model

    elif model_name=="vgg19":
        print(f'model_name :{model_name}')
        print(f"number of epochs :{epochs}")
        print(f"batch size :{batch_size}")
        print(f"image height :{height} width :{width} channels :{channels}")
        if transfer_learing:
            main_model=VGG19(input_shape=(height,width,channels))
            main_model.layers.pop()
            model=tf.keras.Sequential()
            for layer in main_model.layers:
                model.add(layer)
            for layer in main_model.layers:
                layer.trainable=False
            model.add(layers.Dense(num_classes,activation="softmax"))
        else:
            model = VGG19(weights=None ,input_shape=(height,width,channels),classes=classes)
        return model



    elif model_name=="mobilenet_v2":
        print(f'model_name :{model_name}')
        print(f"number of epochs :{epochs}")
        print(f"batch size :{batch_size}")
        print(f"image height :{height} width :{width} channels :{channels}")
        if transfer_learing:
            print("transfer_learning : True ")
            main_model=MobileNet(input_shape=(height,width,channels))
            x=main_model.layers[-6].output
            predictions=layers.Dense(num_classes,activation="softmax")(x)
            model=tf.keras.Model(inputs=main_model.input,outputs=predictions)
            for layer in main_model.layers[:-5]:
                layer.trainable=False
        else:
            model = MobileNetV2( weights=None, input_shape=(height,width,channels),classes=classes)
        return model


    elif model_name=="densenet121":
        model = DenseNet121( weights=None, input_shape=(height,width,channels),classes=classes)
        return model
    elif model_name=="inception_resnet_v2":
        model = InceptionResNetV2( weights=None, input_shape=(height,width,channels),classes=classes)
        return model

