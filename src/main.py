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
num_classes = 50
num_samples = 500
num_epochs = 120
num_batch = 32
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
	print 'label;',label,str(i)+'/'+str(num_classes)
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
	featurewise_center=False,
	featurewise_std_normalization=False,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	horizontal_flip=True,
	dim_ordering='tf')

#%% create and train the model
print 'Training'
model = make_net(num_classes)
#	history = model.fit(X_train, Y_train, verbose=2, validation_split=0.3, \
#	nb_epoch=num_epochs)
#	model.save_weights('weights')

for epoch in range(num_epochs):
	print 'Epoch',str(epoch)+'/'+str(num_epochs)
	batches = 0
	for X_batch, Y_batch in datagen.flow(X_train,Y_train, batch_size=num_batch):
		loss, acc = model.train_on_batch(X_batch,Y_batch)
		vloss, vacc = model.test_on_batch(X_val, Y_val)
		print 'loss:','{0:.3f}'.format(loss),'acc;','{0:.3f}'.format(acc),\
		'val loss:','{0:.3f}'.format(vloss),'val acc;','{0:.3f}'.format(vacc)
		batches += 1
		if batches >= len(X_train)/num_batch:
			break
	if epoch >= save_every and epoch%save_every == 0:
		model.save_weights('m'+str(t.tm_mon)+'_d'+str(t.tm_mday)+'_h'\
		+str(t.tm_hour)+'_e'+str(epoch)+'_weights.hdf5')