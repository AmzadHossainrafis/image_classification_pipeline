# Image_classification_pipeline

# Project overview  
  
 The main goal of this project is to create an image classification pipeline in Keras using several models and to walk you through all of the necessary steps in Keras to create a good and accurate model. This model can be utilized for both binary and multi-class classification. This project explains how to use Keras to train a model, including how to design different types of models, how to integrate transfer learning, how to prepare a data set for training, data augmentation, a data loader/data generator, and how to visualize the results. How to handle and use build in callback functions, Compare the outcomes, learn how to use custom metric, how to reduce the taring time with keras mixed precision and more. In a simple word this pipeline contain all the important aspect of keras for image classification



#project #pipeline #classification #keras 







# Requairments 
 1. Used Modules 
  * [tensorflow-gpu](https://www.tensorflow.org/install/gpu)
  * [keras version](https://pypi.org/project/keras/)
  * [open_cv](https://pypi.org/project/opencv-python/)
  * [Albumentations](https://albumentations.ai/docs/getting_started/installation/)

# Data prepair 
There is a format of the the structure of the directory of the data that must be maintained .For this project we used microsoft cats and dogs data set which can be found in [here](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765)<br>
<p align="center"><img src="logs\dataset.png"\></p>

* Point to be noted individual class must be in separate folder <br>
Exp:-  data/Cats , data/Dogs 

  1. data_split  
You must provide separate data for each machine learning project in order to evaluate the model's performance. The train, test, and validation ratio in most cases is 7:2:1.<br>
All is okay if you downloaded the data(for this project). Just provide the directory( main_dir) in the config.py. If you change the data split as your ratio changes val_ratio, test_ratio from config.py file, then run split() function in the interactive window.
 
  2. data agmentations 

  3. dataloader modification for new class
a simple modification is needed in dataloader.py for new class .Suppose you want to add a new class called Alligator, Just simply add a if condition is the __getitem()
and example is given below and also change num_classes in config.py  <br>
<p align="center"><img src="Screenshot 2022-05-02 172152.png"\></p>


# Training 
  1. model_selection 
  To choose a model simply  just change the variable called model_name from those name or you also implement your own [keras.functional](https://keras.io/guides/functional_api/) model and return it from the model_chg() function 
  
   * list of model implimented in this project from keras.application 
      1. MobilenetV2
      2. VGG16 
      3. VGG19 
      4. Inception_v3
      5. Resnet50
      6. Densenet121<br>
  * list of model implimented in keras.functional 
     1. Alexnet
     
  2. transfer_learning
  * transfer learning is also implimented in vgg16 ,vgg19 , mobilenetv2 , just pass transfer_learning argoment True (model_nam must be vgg16 ,vgg19 or mobilenet_v2)
  
  3. train_the_model
 * To train the model simpley just run the train.py , before that make sure following 
   1. model_nam must be there 
   2. num_classes 
   3. height , width (input shape of alexnet is 227,227,3 all the other model imput shapes are 224,224,3)
   
   
  #### code to train 
    cd dir of the project 
    python train.py
   



