import os
import math
from config import * 
import random
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow import keras
import glob
import albumentations as A
#from tensorflow.keras.utils import to_categorical
from matrice import recall
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def split(dir=data_dir,ratio=val_ratio,test_ratio=test_ratio ,random_state="True"):

    for folder in os.listdir(dir):

        os.makedirs(data_dir+ '/train' + "/"+folder)
        os.makedirs(data_dir + '/val' + "/"+folder)
        os.makedirs(data_dir + '/test' + "/"+folder)
        n=int(len(os.listdir(dir+"/"+folder))*ratio)
        m=int(len(os.listdir(dir+"/"+folder))*test_ratio)
        list1=os.listdir(dir+"/"+folder)
        if random_state=="True":
            for i in range(n):
                random_pick=random.choice(list1)
                os.rename(dir+"/"+folder+"/"+random_pick,data_dir+ '/val' + "/"+folder)
                list1.remove(random_pick)

            for j in range(m):
                random_pick=random.choice(list1)
                os.rename(dir+"/"+folder+"/"+random_pick,data_dir+ '/test' + "/"+folder)
                list1.remove(random_pick)
            for i in list1:   
                os.rename(dir+"/"+folder+"/"+i,data_dir+ '/train' + "/"+folder)

        else:
            for i in range(n):
                os.rename(os.listdir(dir+"/"+folder)[i],data_dir+ '/val' + "/"+folder)

            for j in range(m):
                os.rename(dir+"/"+folder+"/"+j,data_dir+ '/test' + "/"+folder)
            
            for i in os.listdir(dir+"/"+folder):
                os.rename(os.listdir(dir+"/"+folder)[i],data_dir+ '/train' + "/"+folder)      


        
def prediction(model_name=model_name,model_dir=load_model_path,randoms=True,rangeS=None,height=height,width=width):
    cat_dir = glob.glob(glob_file1)
    dog_dir = glob.glob(glob_file2)
    train_dir = (cat_dir + dog_dir)
    model=keras.models.load_model(model_dir,custom_objects={'recall':recall})

    if randoms==True:
        reandom_choice=random.randint(1,len(train_dir))
        img = keras.preprocessing.image.load_img(train_dir[reandom_choice], target_size=(height, width))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.resize(x, (1, height, width, 3))
        pred=model.predict(x)
        plt.title("random image")
        plt.imshow(img)
        plt.xlabel(pred)
        plt.savefig(prediction_dir+"\\"+ model_name+"_prediction{}.png".format(reandom_choice))
        plt.show()
    if rangeS is not None:
        row=3
        col=math.floor(rangeS/row)
        fig , axes = plt.subplots(row,col)
        for i in range(row):
            for j in range(col):
                
                reandom_choice=random.choice(train_dir)
                img = keras.preprocessing.image.load_img(reandom_choice, target_size=(height, width))
                x = keras.preprocessing.image.img_to_array(img)
                x = np.resize(x, (1, height, width, 3))
                pred=model.predict(x)
                axes[i][j].imshow(img)  
                axes[i][j].set_title(pred)
                axes[i][j].axis('off')
                

                

               
def confusionmatrix():
    cat_dir = glob.glob(glob_file1)
    dog_dir = glob.glob(glob_file2)
    train_dir = (cat_dir + dog_dir)
    y=[]
    predictions=[]
    model=keras.models.load_model(load_model_path,custom_objects={'recall':recall})
    for path in train_dir:
        img = keras.preprocessing.image.load_img(path, target_size=(height, width))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.resize(x, (1, height, width, 3))
        x=model.predict(x)
       
        if x[0][0]>x[0][1]:
            predictions.append(0)   
        else:    
            predictions.append(1)

        name = path.split("/")[-1].split("\\")[-2]
        if name == "Cat":
        
            y.append(1)
        else:
            y.append(0)
    cm=confusion_matrix(y,predictions,labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.show()
   

    

def agment():
    transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

    
    
def all_comparison(col,csv_dir):
    pass

    






