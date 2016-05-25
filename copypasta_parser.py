# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:51:04 2016

@author: ozer
"""

s = """
21s - loss: 1.6151 - acc: 0.2088 - val_loss: 1.6212 - val_acc: 0.2000
Epoch 2/120
20s - loss: 1.6200 - acc: 0.1924 - val_loss: 1.6131 - val_acc: 0.2000
Epoch 3/120
20s - loss: 1.6173 - acc: 0.1888 - val_loss: 1.6125 - val_acc: 0.2000
Epoch 4/120
20s - loss: 1.6147 - acc: 0.1964 - val_loss: 1.6143 - val_acc: 0.2000
Epoch 5/120
20s - loss: 1.6156 - acc: 0.1968 - val_loss: 1.6106 - val_acc: 0.2000
Epoch 6/120
20s - loss: 1.6141 - acc: 0.2048 - val_loss: 1.6112 - val_acc: 0.2000
Epoch 7/120
20s - loss: 1.6153 - acc: 0.1876 - val_loss: 1.6193 - val_acc: 0.2000
Epoch 8/120
20s - loss: 1.6193 - acc: 0.1896 - val_loss: 1.6100 - val_acc: 0.2000
Epoch 9/120
20s - loss: 1.6158 - acc: 0.2036 - val_loss: 1.6156 - val_acc: 0.2000
Epoch 10/120
21s - loss: 1.6164 - acc: 0.1940 - val_loss: 1.6145 - val_acc: 0.2000
Epoch 11/120
20s - loss: 1.6168 - acc: 0.1960 - val_loss: 1.6144 - val_acc: 0.2000
Epoch 12/120
20s - loss: 1.6145 - acc: 0.1972 - val_loss: 1.6144 - val_acc: 0.2000
Epoch 13/120
20s - loss: 1.6165 - acc: 0.1992 - val_loss: 1.6154 - val_acc: 0.2000
Epoch 14/120
21s - loss: 1.6168 - acc: 0.2008 - val_loss: 1.6208 - val_acc: 0.2000
Epoch 15/120
20s - loss: 1.6169 - acc: 0.1828 - val_loss: 1.6117 - val_acc: 0.2000
Epoch 16/120
20s - loss: 1.6165 - acc: 0.1924 - val_loss: 1.6157 - val_acc: 0.2000
Epoch 17/120
20s - loss: 1.6149 - acc: 0.1980 - val_loss: 1.6199 - val_acc: 0.2000
Epoch 18/120
20s - loss: 1.6169 - acc: 0.2012 - val_loss: 1.6148 - val_acc: 0.2000
Epoch 19/120
20s - loss: 1.6128 - acc: 0.2084 - val_loss: 1.6187 - val_acc: 0.2000
Epoch 20/120
20s - loss: 1.6152 - acc: 0.1956 - val_loss: 1.6163 - val_acc: 0.2000
Epoch 21/120
20s - loss: 1.6144 - acc: 0.1996 - val_loss: 1.6144 - val_acc: 0.2000
Epoch 22/120
20s - loss: 1.6177 - acc: 0.1952 - val_loss: 1.6162 - val_acc: 0.2000
Epoch 23/120
20s - loss: 1.6169 - acc: 0.1992 - val_loss: 1.6154 - val_acc: 0.2000
Epoch 24/120
20s - loss: 1.6149 - acc: 0.1996 - val_loss: 1.6130 - val_acc: 0.2000
Epoch 25/120
20s - loss: 1.6141 - acc: 0.1996 - val_loss: 1.6121 - val_acc: 0.2000
Epoch 26/120
21s - loss: 1.6141 - acc: 0.1976 - val_loss: 1.6147 - val_acc: 0.2000

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



