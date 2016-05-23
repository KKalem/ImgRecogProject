# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:09:26 2016

@author: ozer
"""
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

Convrelu = lambda num_filt,filt_size: Convolution2D(num_filt, filt_size, filt_size,\
dim_ordering = 'tf', border_mode = 'valid', init = 'glorot_normal', activation='relu')

Conv = lambda num_filt, filt_size: Convolution2D(num_filt, filt_size, filt_size,\
dim_ordering = 'tf', border_mode = 'valid', init = 'glorot_normal')

Zeropad = lambda size=1: ZeroPadding2D(padding = (size,size), dim_ordering = 'tf')

Maxpool = lambda size=2: MaxPooling2D((size,size), strides=(size,size))

Relu = lambda size: Dense(size, activation='relu', init='glorot_normal')

def make_net(num_classes):
	model = Sequential()
	model.add(ZeroPadding2D(padding = (2,2), dim_ordering='tf',input_shape=(64,64,3)))
	model.add(Convrelu(32,3))
	model.add(Zeropad())
	model.add(Convrelu(32,3))
	model.add(Maxpool())
	model.add(Dropout(0.25))

#	model.add(Zeropad())
#	model.add(Convrelu(64,3))
#	model.add(Convrelu(64,3))
#	model.add(Maxpool())
#	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Relu(128))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optim, loss='categorical_crossentropy', metrics=['accuracy'])
	return model