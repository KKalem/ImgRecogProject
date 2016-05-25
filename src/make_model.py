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

Maxpool = lambda size=2, stride=2: MaxPooling2D((size,size), strides=(stride,stride))

Relu = lambda size: Dense(size, activation='relu', init='glorot_normal')

def make_net(num_classes):
	model = Sequential()
	model.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu',\
	border_mode='same',dim_ordering='tf'))

	model.add(Convolution2D(32,3,3, activation='relu', border_mode='same',dim_ordering='tf'))

	model.add(Maxpool())

	model.add(Flatten())
	model.add(Relu(128))
	#model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optim, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
