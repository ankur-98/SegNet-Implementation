#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 17:47:45 2018

@author: ankur
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
import numpy as np
import cv2
from tqdm import tqdm
import pickle as pk
from copy import deepcopy,copy

import keras.backend as K
import tensorflow as tf
K.set_image_data_format('channels_last')
from keras.layers import Input,Flatten,BatchNormalization,Dense,AveragePooling2D,MaxPooling2D,Reshape,MaxPooling3D,AveragePooling3D,Activation
from keras.layers import UpSampling2D,Conv2D,Conv2DTranspose,Add
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.models import Model
from keras.optimizers import Adam, Adadelta, RMSprop, Nadam, Adamax
import datetime
from keras.losses import categorical_crossentropy
from keras import callbacks
from keras.applications.vgg16 import VGG16

main_dir = '/Seg'
data_dir = '/data/DAVIS_seg_data'
data_list = '/data/DAVIS_seg_data/ImageSets'
resol = {480: '480p',1080:'1080p'}


class DATA:
    
    def __init__(self,batch_size=200):
        self.resoln = resol[480]
           
       x_train_loc, y_train_loc = self.get_loc('train.txt')
       x_val_loc, y_val_loc = self.get_loc('val.txt')    
       train_X,train_Y = self.get_nparray(x_train_loc,y_train_loc)
       val_X,val_Y = self.get_nparray(x_val_loc,y_val_loc)
       pk.dump([train_X,val_X,train_Y,val_Y], open(os.path.join(data_dir,"data.p"), "wb"), protocol=4)
       print("\nPickle File Created===============================================================================\n")
       input("Do you wish to continue? If Yes press enter\n")
    
        with open(os.path.join(data_dir,"data.p"), "rb") as p:
            self.train_x, self.val_x, self.train_y, self.val_y = pk.load(p)
            print("\nPickle File Loaded================================================================================\n")

        self.height, self.width = self.train_x.shape[1:3]
        self.batch_size = {'Train':batch_size, 'Val':batch_size, 'Test':batch_size}
        
    def get_loc(self,PATH):
        with open(os.path.join(data_list,self.resoln,PATH)) as f:
            x_loc = []
            y_loc = []
            for line in f:
                x_loc.append(data_dir + line.split()[0])
                y_loc.append(data_dir + line.split()[1])
        return np.array(x_loc),np.array(y_loc)
    
    def create_2_channel(self,x):
        x0 = np.expand_dims(np.array(x[...,0]/255,dtype=np.uint8),axis=2)
        x0_ = np.add(np.invert(x0),2)
        x_ = np.append(x0,x0_,axis=2)
        return x_
    
    def get_nparray(self,X,Y):
        npX = []
        npY = []
        for x,y in tqdm(zip(X,Y)):
            npX.append(cv2.imread(x))
            npY.append(self.create_2_channel(cv2.imread(y)))
        npX = np.pad(npX,[(0,0),(0,0),(0,960-np.array(npX).shape[2]),(0,0)],mode='constant')
        npY = np.pad(npY,[(0,0),(0,0),(0,960-np.array(npY).shape[2]),(0,0)],mode='constant')
        return npX,npY
    

def customSegLoss_PixelWise(yTrue,yPred):
    
    fg = tf.reduce_sum(yTrue)/1.0
    bg = tf.to_float(tf.size(yTrue)) - fg
    
    w = fg/tf.to_float(tf.size(yTrue))

    fg_i = K.cast(K.equal(yTrue,1),K.floatx())
    bg_i = K.cast(K.equal(yTrue,0),K.floatx())
    
    Loss_fg = K.sum(K.log(K.clip(fg_i * yPred,K.epsilon(),None)))
    Loss_bg = K.sum(K.log(K.clip(bg_i * yPred,K.epsilon(),None)))
    
    Loss = (Loss_fg * (w-1)) + (Loss_bg * (-w))
    
    
    return Loss

def metric_iou(yTrue, yPred):
    
    y_ = K.cast(K.greater_equal(yPred,0.5),tf.int8)
    y = K.cast(K.equal(yTrue,1),tf.int8)
    
    I = K.sum(y * y_)
    U = K.sum(y) + K.sum(y_) - I
    
    return K.switch(K.equal(U, 0), 1.0, I / U)
    


def get_model(d,model_param):
    
    m = VGG16(include_top=False, weights='imagenet', input_shape=(d.train_x.shape[1:]))
    for l in m.layers[:]:
        l.trainable = model_param['trainable']
    x = m.output
    
    block1_MaxPoolingInput,block2_MaxPoolingInput,block3_MaxPoolingInput,block4_MaxPoolingInput,block5_MaxPoolingInput = [m.layers[3].input,m.layers[6].input,m.layers[10].input,m.layers[14].input,m.layers[18].input]
    
    # Block 5
    x = UpSampling2D((2,2),name='block5_Upsampled')(x)
    x = Add()([x, block5_MaxPoolingInput])
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv5')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv6')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Block 4
    x = UpSampling2D((2,2),name='block4_Upsampled')(x)
    x = Add()([x, block4_MaxPoolingInput])
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv5')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv6')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Block 3
    x = UpSampling2D((2,2),name='block3_Upsampled')(x)
    x = Add()([x, block3_MaxPoolingInput])
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv5')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv6')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Block 2
    x = UpSampling2D((2,2),name='block2_Upsampled')(x)
    x = Add()([x, block2_MaxPoolingInput])
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv5')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Block 1
    x = UpSampling2D((2,2),name='block1_Upsampled')(x)
    x = Add()([x, block1_MaxPoolingInput])
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv4')(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2, (3, 3), padding='same', name='block1_conv5')(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)
    
    model = Model(m.input,x)
    
    optim = Adam()
#    optim = Adadelta()
#    optim = RMSprop()
#    optim = Nadam()
#    optim = Adamax()    
    model.compile(optimizer=optim, loss=customSegLoss_PixelWise, metrics=['accuracy',metric_iou])
    
    model.summary()
    return model
    
def save_model(model,d,DT):
    # serialize model to JSON
    model_json = model.to_json()
    model_name = "model_" + str(DT)
    with open(os.path.join(main_dir,"model",model_name+".json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(main_dir,"model",model_name+".h5"))
    pk.dump(d,open(os.path.join(main_dir,"model","d_"+DT+".pk"), "wb"),protocol=pk.HIGHEST_PROTOCOL)
    print("\nCheckpoints Created===============================================================================\n")
    print("\nModel Files Created===============================================================================\n")


if __name__ == '__main__':

    model_param = {'batch_size':4, 'epoochs': 100, 'trainable':False}
    
    d = DATA(model_param['batch_size'])
    print("\nData Loaded =====================================================================================\n")
    
    
    model = get_model(d,model_param)
    print("\nModel Loaded =====================================================================================\n")

##################################################################################################################
# Callbacks
##################################################################################################################
    DT = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    checkp = os.path.join(main_dir,'model_checkpoint',DT)

    checkpoint1 = callbacks.ModelCheckpoint(checkp + "_{epoch:02d}_train_loss.hdf5",
                                           monitor='loss', verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1)

    checkpoint2 = callbacks.ModelCheckpoint(checkp + "_{epoch:02d}_train_acc.hdf5",
                                            monitor='acc', verbose=0, save_best_only=True,
                                            save_weights_only=False, mode='auto', period=1)
    
    checkpoint3 = callbacks.ModelCheckpoint(checkp + "_{epoch:02d}_loss.hdf5",
                                           monitor='val_loss', verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1)

    checkpoint4 = callbacks.ModelCheckpoint(checkp + "_{epoch:02d}_acc.hdf5",
                                            monitor='val_acc', verbose=0, save_best_only=True,
                                            save_weights_only=False, mode='auto', period=1)
    
    checkpoint5 = callbacks.ModelCheckpoint(checkp + "_{epoch:02d}_train_metric_iou.hdf5",
                                           monitor='iou', verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1)

    checkpoint6 = callbacks.ModelCheckpoint(checkp + "_{epoch:02d}_val_metric_iou.hdf5",
                                            monitor='val_iou', verbose=0, save_best_only=True,
                                            save_weights_only=False, mode='auto', period=1)

    tfboard = callbacks.TensorBoard(log_dir=checkp,
                                    histogram_freq=0, batch_size=d.batch_size['Train'],
                                    write_graph=True, write_grads=True, write_images=True,
                                    embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
                
    callback = [checkpoint1, checkpoint2, checkpoint3, checkpoint4, checkpoint5, checkpoint6, tfboard]

##################################################################################################################

##################################################################################################################
# Training
##################################################################################################################
    
    print("\nStarting to train! ====================================================================================\n")

    from keras.backend.tensorflow_backend import set_session, get_session

    s = get_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    s._config = config
    set_session(s)  # set this TensorFlow session as the default session for Keras
#    s.run()

    try:
        model.fit(x=d.train_x, y=d.train_y, batch_size=d.batch_size['Train'], 
                  epochs=model_param['epoochs'], callbacks=callback, validation_data=(d.val_x,d.val_y), 
                  shuffle=True, steps_per_epoch=None, validation_steps=None)
    finally:
        if input('want to save model: ') == 'y':
           save_model(model,d,DT)
           
    print("\nTraining Ended====================================================================================\n")
##################################################################################################################
##################################################################################################################
