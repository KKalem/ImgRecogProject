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
import time

t = time.localtime()
num_classes = 10
num_samples = 500
num_epochs = 120
num_batch = 64
save_every = 20

trainpath = '../tiny-imagenet-200/train/'
label_names = os.listdir(trainpath)
label_encodings = np.eye(num_classes, dtype=np.int)
label_map = dict(zip(label_names,label_encodings))

#%% load the training images
print 'Generating X_train'
X = []
Y = []
for i in range(num_classes):
	label = label_names[i]
	print 'label;',label,str(i+1)+'/'+str(num_classes)
	im_names = os.listdir(trainpath+label+'/images')
	for s in range(num_samples):
		im_name = im_names[s]
		im_path = trainpath+label+'/images/'+im_name
		im = imio.imread(fname = im_path, as_grey = False)
		if im.shape != (64,64,3): #some images are grey, stack'em
			im = np.dstack([im,im,im])
		X.append(im/255.) #fit into 0-1
		Y.append(label_encodings[i])
X_train = np.array(X)
Y_train = np.array(Y)
#%% read the val text file to see which image belongs to which class
valpath = '../tiny-imagenet-200/val/'
raw = np.loadtxt(valpath+'val_annotations.txt', dtype=np.str, delimiter='\t')
valimagepath = valpath+'images/'

filenames = raw[:,0]
labelnames = raw[:,1]
val_label_encodings = []
for labelname in labelnames:
	if labelname in label_map:
		label_encoding = label_map.get(labelname)
		val_label_encodings.append(label_encoding)
#encoding of filenames to class vectors
val_label_map = dict(zip(filenames,val_label_encodings))
#%% load the validation set images and label them
print 'Generating X_val'
X = []
Y = []

images = os.listdir(valimagepath)
for image in images:
	if image in val_label_map:
		im = imio.imread(fname = valimagepath+image, as_grey = False)
		if im.shape != (64,64,3):
			im = np.dstack([im,im,im])
		X.append(im/255.)
		Y.append(val_label_map.get(image))

X_val = np.array(X)
Y_val = np.array(Y)

del X
del Y
#%% generator to create rotated, shifted etc images for training
datagen = ImageDataGenerator(
#	featurewise_center = False,
#	featurewise_std_normalization = False,
#	zca_whitening = False,
#	fill_mode='constant',
#	cval=0,
#	rotation_range=0,
#	width_shift_range=0.,
#	height_shift_range=0.,
#	horizontal_flip=False,
	dim_ordering='tf')


#print 'fitting datagen'
#datagen.fit(X_train)
#%% create and train the model
print 'Training'
model = make_net(num_classes)
#%%
#history = model.fit(X_train, Y_train, verbose=2, validation_data=(X_val, Y_val),
#nb_epoch=num_epochs)
#model.save_weights('weights')

#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=num_batch),
#					samples_per_epoch=len(X_train), nb_epoch=num_epochs,
#					validation_data=(X_val, Y_val))

for epoch in range(num_epochs):
	print 'Epoch',str(epoch+1)+'/'+str(num_epochs)
	start = time.time()
	batches = 0
	tloss, tacc = 0. , 0.
	for X_batch, Y_batch in datagen.flow(X_train,Y_train, batch_size=num_batch):
#		loss, acc = model.train_on_batch(X_batch,Y_batch)
		history = model.fit(X_batch, Y_batch, batch_size=len(X_batch), nb_epoch=1, verbose=0)
		loss = history.history['loss'][0]
		acc = history.history['acc'][0]
		tloss += loss
		tacc += acc
		batches += 1
		if batches >= len(X_train)/num_batch:
			break
	tloss = tloss/batches
	tacc = tacc/batches
#	vloss, vacc = model.evaluate(X_val, Y_val, batch_size=num_batch, verbose=0)
	vloss, vacc = model.evaluate_generator(datagen.flow(X_val, Y_val, batch_size=num_batch),\
	val_samples = len(X_val))
	end = time.time()
	print '{0:.3f}s'.format(end-start),\
		'loss:','{0:.9f}'.format(tloss),\
		'acc;','{0:.9f}'.format(tacc),\
		'val_loss:','{0:.9f}'.format(vloss),\
		'val_acc;','{0:.9f}'.format(vacc)
	if epoch >= save_every and epoch%save_every == 0:
		model.save_weights('m'+str(t.tm_mon)+'_d'+str(t.tm_mday)+'_h'\
		+str(t.tm_hour)+'_e'+str(epoch)+'_weights.hdf5')