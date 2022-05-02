# Image_classification_pipeline

# Project overview 
  1. Intruduction 
The main goal of this project is to create an image classification pipeline in Keras using several models and to walk you through all of the necessary steps in Keras to create a good and accurate model. This model can be utilized for both binary and multi-class classification. This project explains how to use Keras to train a model, including how to design different types of models, how to integrate transfer learning, how to prepare a data set for training, data augmentation, a data loader/data generetor, and how to visualize the results.how to handel and use build in callback fucntions, Compare the outcomes, learn how to use custom metric, how to reduce the traing time with keras mixed precision and more. In a simple word this pipeline contain all the importent aspet of keras for imagae clssification 


# Requairments 
 1. Used Modules 
  * [tensorflow-gpu](https://www.tensorflow.org/install/gpu)
  * [keras version](https://pypi.org/project/keras/)
  * [open_cv](https://pypi.org/project/opencv-python/)

# Data prepair 
There is a format of the the structure of the directory of the data that must be maintained .For this project we used microsoft cats and dogs data set which can be found in [here](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765)<br>

* Point to be noted individual class must be in separate folder 
Exp:-  data/Cats , data/Dogs 

  1. data_split  
You must provide separate data for each machine learning project in order to evaluate the model's performance. The train, test, and validation ratio in most cases is 7:2:1.<br>
All is okay if you downloaded the data(for this project). Just provide the directory( main_dir) in the config.py. If you change the data split as your ratio changes val_ratio, test_ratio from config.py file, then run split() function in the interactive window.
 
  2. data agmentations 
  3. dataloader modification for new class
# Training 
  1. model_selection 
  2. transfer_learning 
  3. train_the_model
  4. how_to_train_custom_model 
# Result 
  1.how to predict 
  2.model outcome 
  
# Result compare 
  1.how to use confusion matrices 
  2. compare recalls f-1 

# Utils functions overview 


