'''
Created on Apr. 23, 2019

    simple template for keras train
    
@author: deisler
'''

import os
from dataset.data_generator import trainGenerator

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from segmentation_models import *
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

NO_OF_TRAINING_IMAGES = len(os.listdir('/your_data/train_frames/train/'))
NO_OF_VAL_IMAGES = len(os.listdir('/your_data/val_frames/val/'))

NO_OF_EPOCHS = 'ANYTHING FROM 30-100 FOR SMALL-MEDIUM SIZED DATASETS IS OKAY'

BATCH_SIZE = 'BATCH SIZE PREVIOUSLY INITIALISED'

weights_path = 'path/where/resulting_weights_will_be_saved'

aug_dict = {rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True}

train_path = '/your_data/train_frames/train/'
val_path = '/your_data/train_frames/val/'
# image folder with class names
image_folder = ''
# 
mask_folder = ''
# train data save path
save_dir = ''

seed = 134

train_gen = trainGenerator(BATCH_SIZE,train_path,image_folder,mask_folder,aug_dict,num_class = 2,save_to_dir = save_dir, seed = seed)

val_gen = trainGenerator(BATCH_SIZE,val_path,image_folder,mask_folder,aug_dict,num_class = 2,save_to_dir = save_dir, seed = seed)

# define model
model = Unet(BACKBONE, encoder_weights='imagenet')


opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

'''
m.compile(loss='The loss to optimise [eg: dice_loss],
              optimizer=opt,
              metrics='YOUR_METRIC [could be 'accuracy' or mIOU, dice_coeff etc]')
'''
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])


checkpoint = ModelCheckpoint(weights_path, monitor='METRIC_TO_MONITOR', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'METRIC_TO_MONITOR', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

callbacks_list = [checkpoint, csv_logger, earlystopping]

results = model.fit_generator(train_gen, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          callbacks=callbacks_list)
model.save('Model.h5')