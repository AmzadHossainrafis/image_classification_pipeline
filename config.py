


main_dir = r'C:\Users\Amzad\Desktop\keras_project\image_classification'

data_dir = main_dir + '\data'
val_ratio = 0.2
test_ratio = 0.4
train_dir = data_dir + '\train'
val_dir = data_dir + '\val'
test_dir = data_dir + '\\test'


model_name="mobilenet_v2"
batch_size = 16
num_classes = 2
epochs = 20
step_size=10

height = 224
width=224
channels=3
num_classes=2

log_dir=r"C:\Users\Amzad\Desktop\keras_project\image_classification\project\logs"

csv_log_dir=log_dir+"\csv_log"
tensorboard_log_dir=log_dir+"\tensorboard_log"
weights_dir=log_dir+"\weights"
early_stoping=False
glob_file1="C:/Users/Amzad/Desktop/keras_project/image_classification/data/test/Cat/*.*"
glob_file2="C:/Users/Amzad/Desktop/keras_project/image_classification/data/test/Dog/*.*"
prediction_dir=r"C:\Users\Amzad\Desktop\keras_project\image_classification\prediction"

load_model_path=r'C:\Users\Amzad\Desktop\keras_project\image_classification\project\logs\weights\weights_of_vgg19_date_2022-04-24_15-37-55.hdf5'