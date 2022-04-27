
from dataloader import train_ds, val_ds
from model import model_chg
from config import * 
import os
import datetime
import tensorflow as tf
from tensorflow import keras
from matrice import recall
#from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"]="0"

model=model_chg(model_name=model_name,transfer_learing=True,height=height,width=width,channels=channels,classes=num_classes)

chk_point=keras.callbacks.ModelCheckpoint(weights_dir+"/weights_of_{}_date_{}.hdf5".format(model_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
csv_logger=keras.callbacks.CSVLogger(csv_log_dir+"/{}_{}.csv".format(model_name,datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"),separator=",",append=False))

early_stoping=keras.callbacks.EarlyStopping(monitor="val_loss",patience=5)




callback = [
    chk_point,csv_logger,early_stoping
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",recall,keras.metrics.Precision()],
)





model.fit(
    train_ds, epochs=epochs, callbacks=callback, validation_data=val_ds,
)

