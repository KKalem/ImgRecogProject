# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:51:04 2016

@author: ozer
"""

s = """
224s - loss: 4.6136 - acc: 0.0101 - val_loss: 4.6132 - val_acc: 0.0100
Epoch 2/120
225s - loss: 4.6138 - acc: 0.0092 - val_loss: 4.6132 - val_acc: 0.0100
Epoch 3/120
226s - loss: 4.6137 - acc: 0.0099 - val_loss: 4.6119 - val_acc: 0.0100
Epoch 4/120
226s - loss: 4.6136 - acc: 0.0095 - val_loss: 4.6121 - val_acc: 0.0100
Epoch 5/120
224s - loss: 4.6138 - acc: 0.0099 - val_loss: 4.6114 - val_acc: 0.0100
Epoch 6/120
218s - loss: 4.6138 - acc: 0.0100 - val_loss: 4.6097 - val_acc: 0.0100
Epoch 7/120
219s - loss: 4.6136 - acc: 0.0091 - val_loss: 4.6108 - val_acc: 0.0100
Epoch 8/120
219s - loss: 4.6138 - acc: 0.0096 - val_loss: 4.6126 - val_acc: 0.0100
Epoch 9/120
219s - loss: 4.6138 - acc: 0.0099 - val_loss: 4.6111 - val_acc: 0.0100
Epoch 10/120
220s - loss: 4.6136 - acc: 0.0101 - val_loss: 4.6125 - val_acc: 0.0100
Epoch 11/120
219s - loss: 4.6136 - acc: 0.0095 - val_loss: 4.6119 - val_acc: 0.0100
Epoch 12/120
219s - loss: 4.6132 - acc: 0.0096 - val_loss: 4.6116 - val_acc: 0.0100
Epoch 13/120
220s - loss: 4.6138 - acc: 0.0099 - val_loss: 4.6103 - val_acc: 0.0100
"""
#%%
tokens = s.split(' ')
stokens = []
for token in tokens:
	stokens.extend(token.split('\n'))
losses = []
accs = []
val_losses=[]
for i in range(1,len(tokens)):
	prev = stokens[i-1]
	if prev == 'loss:':
		losses.append(float(stokens[i]))
	if prev == 'acc:':
		accs.append(float(stokens[i]))
	if prev == 'val_loss:':
		val_losses.append(float(stokens[i]))



