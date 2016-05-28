# -*- coding: utf-8 -*-
"""
Created on Sat May 28 11:15:02 2016

@author: ozer
"""

import os
import numpy as np
import skimage.io as imio
from make_model import make_net
from keras.preprocessing.image import ImageDataGenerator
import time
import vgg_19 as vgg
from keras.optimizers import SGD
from keras import backend as K

t = time.localtime()
num_classes = 2

trainpath = '../tiny-imagenet-200/train/'
label_names = os.listdir(trainpath)
label_encodings = np.eye(num_classes, dtype=np.int)
label_map = dict(zip(label_names,label_encodings))


model = vgg.VGG_19('vgg19_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
#%%
label = 'n01443537'
im_name = 'n01443537_0.JPEG'
im_path = trainpath+label+'/images/'+im_name
im = imio.imread(fname = im_path, as_grey = False)
#%%
def convert_to_th(im):
	thim = np.array([im[:,:,0],im[:,:,1],im[:,:,2]])
	return thim

def convert_to_tf(im):
	tfim = np.dstack([im[0,:,:],im[1,:,:],im[2,:,:]])
	return tfim

#%%
get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[22].output])
thim = convert_to_th(im)
base = np.zeros((3,224,224))
base[:,0:64,0:64] = thim[:,:,:]

out = get_layer_output([np.array([base])])[0][0]
imio.imshow(out[1,:64,:64])