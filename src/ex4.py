# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:30:55 2016

@author: ozer
"""
#%%
import numpy as np
import scipy.io as sio
import cv2

from keras.optimizers import SGD
from keras import backend as K

import vgg_19 as vgg #this is the convnet
import ex3 as svd #ex3 is the file from exercise 3

#%%
#this part is only used to save the extracted features to file
def extract(matlabfile,outfile):
	res_ims = []
	with open(matlabfile) as matfile:
		mat = sio.loadmat(matfile)
#		ys = mat['labels']
		ims = mat['ims'][0]
		for im in ims:
			im = cv2.resize(im, (224, 224)).astype(np.float32)
			im[:,:,0] -= 103.939
			im[:,:,1] -= 116.779
			im[:,:,2] -= 123.68
			im = im.transpose((2,0,1))
			im = np.expand_dims(im, axis=0)
			res_ims.append(im)
	#%%
	layer_outputs = []
	model = vgg.VGG_19('vgg19_weights.h5')
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) #not used since the weights are pre-trained
	model.compile(optimizer=sgd, loss='categorical_crossentropy')

	#get_37th_layer_output = K.function([model.layers[0].input], [model.layers[38].output]) #37 = flatten
	get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[41].output]) #41 = last layer-1
	i = 0
	l = '('+str(len(res_ims))+')'
	for im in res_ims:
		print 'progress',str(i)+l
		i += 1
	#	layer_output = get_37th_layer_output([im])[0][0]
		layer_output = get_layer_output([im, 1])[0][0]
		layer_outputs.append(layer_output)

	Xtrain = np.array(layer_outputs).T
	np.save(outfile,Xtrain)
#%%
#####################################################################################################
#####################################################################################################
#####################################################################################################
if __name__ == '__main__':
	extract('inria_train.mat','Xtrain')
	with open('inria_train.mat') as matfile:
		mat = sio.loadmat(matfile)
		y = mat['labels']
	Xtrain = np.load('Xtrain.npy')
	w,b,m,v = svd.train_SVD(Xtrain,y)

	extract('inria_test.mat','Xtest')
	with open('inria_test.mat') as matfile:
		mat = sio.loadmat(matfile)
		y_t = mat['labels']
	Xtest = np.load('Xtest.npy')

	predictions = svd.classify_SVD(Xtest,v,m,y_t)




