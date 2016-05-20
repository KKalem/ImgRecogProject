# -*- coding: utf-8 -*-
"""
Created on Tue May 17 19:49:10 2016

@author: ozer
"""
import os
import numpy as np
import skimage.io as imio
from make_model import make_net
from keras.preprocessing.image import ImageDataGenerator
try:
	num_classes = 4
	num_samples = 100
	num_epochs = 100

	trainpath = 'tiny-imagenet-200/train/'
	label_names = os.listdir(trainpath)
	label_encodings = np.eye(num_classes, dtype=np.int)
	label_map = dict(zip(label_names,label_encodings))

	#%%
	X = []
	Y = []
	for i in range(num_classes):
		label = label_names[i]
		print 'label;',label
		im_names = os.listdir(trainpath+label+'/images')
		for s in range(num_samples):
			im_name = im_names[s]
			im_path = trainpath+label+'/images/'+im_name
			im = imio.imread(fname = im_path, as_grey = False)
			if im.shape != (64,64,3):
	#			print 'grayscale;',i,s,im.shape,im_path
				im = np.dstack([im,im,im])
			X.append(im/255.)
			Y.append(label_encodings[i])
	X_train = np.array(X)
	Y_train = np.array(Y)
	#%% THIS IS STUPID, DONT DO THIS, THIS ONLY DOES THESE FOR ALL THE DATA ONCE, STATIC, STUPID ETC
	datagen = ImageDataGenerator(
		featurewise_center=False,
		featurewise_std_normalization=False,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		dim_ordering='tf')

#	flow = datagen.flow(X_train,Y_train, batch_size=len(X_train))
#	n = flow.next()
#	X_train = n[0]
#	Y_train = n[1]
	#%%
	model = make_net(num_classes)
	history = model.fit(X_train, Y_train, verbose=2, validation_split=0.3, nb_epoch=num_epochs)
	model.save_weights('weights')
finally:
	print 'finally'
	model.save_weights('weights')
