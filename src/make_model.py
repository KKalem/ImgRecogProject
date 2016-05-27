# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:09:26 2016

@author: ozer
"""
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.regularizers import WeightRegularizer
from keras.layers.normalization import BatchNormalization

def make_net(num_classes):
	model = Sequential()

	model.add(ZeroPadding2D((2,2), input_shape = (64,64,3), dim_ordering = 'tf'))
	model.add(Convolution2D(64,5,5, sumsample = (2,2),\
	W_regularizer = WeightRegularizer(l1=1e-1,l2=1e-1),dim_ordering = 'tf'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))


	model.add(ZeroPadding2D((1,1), dim_ordering = 'tf'))
	model.add(Convolution2D(64,3,3, sumsample = (1,1),\
	W_regularizer = WeightRegularizer(l1=1e-1,l2=1e-1), dim_ordering = 'tf'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))


	model.add(ZeroPadding2D((1,1), dim_ordering = 'tf'))
	model.add(Convolution2D(128,3,3, sumsample = (2,2), dim_ordering = 'tf'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))


	model.add(Flatten())
	model.add(Dense(512, W_regularizer = WeightRegularizer(l1=1e-2,l2=1e-2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))



#	model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(64,64,3), dim_ordering='tf'))
#
#
#	model.add(Flatten())
#	model.add(Dense(num_classes))
#	model.add(Activation('softmax'))

	optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optim, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
