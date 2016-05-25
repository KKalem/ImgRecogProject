# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:51:04 2016

@author: ozer
"""

s = """
Epoch 1/120
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 4392 get requests, put_count=4331 evicted_count=1000 eviction_rate=0.230894 and unsatisfied allocation rate=0.264344
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 100 to 110
250/250 [==============================] - 0s
23.401s loss; 1.623798361 acc; 0.196714744 val_loss; 1.612657193 val_acc; 0.200000000
Epoch 2/120
250/250 [==============================] - 0s
22.617s loss; 1.618067932 acc; 0.197115385 val_loss; 1.612050258 val_acc; 0.200000000
Epoch 3/120
250/250 [==============================] - 0s
22.446s loss; 1.618047121 acc; 0.190705128 val_loss; 1.614283461 val_acc; 0.200000000
Epoch 4/120
250/250 [==============================] - 0s
22.568s loss; 1.617090957 acc; 0.197115385 val_loss; 1.615766908 val_acc; 0.200000000
Epoch 5/120
250/250 [==============================] - 0s
22.570s loss; 1.614266267 acc; 0.208333333 val_loss; 1.610156164 val_acc; 0.200000000
Epoch 6/120
250/250 [==============================] - 0s
22.779s loss; 1.614304605 acc; 0.213942308 val_loss; 1.618192377 val_acc; 0.200000000
Epoch 7/120
250/250 [==============================] - 0s
22.830s loss; 1.615733163 acc; 0.202724359 val_loss; 1.614787613 val_acc; 0.200000001
Epoch 8/120
250/250 [==============================] - 0s
22.918s loss; 1.613916260 acc; 0.205528846 val_loss; 1.617527492 val_acc; 0.200000001
Epoch 9/120
250/250 [==============================] - 0s
22.791s loss; 1.616669045 acc; 0.199519231 val_loss; 1.615328062 val_acc; 0.200000000
Epoch 10/120
250/250 [==============================] - 0s
22.924s loss; 1.614657092 acc; 0.208333333 val_loss; 1.612100225 val_acc; 0.200000000
Epoch 11/120
250/250 [==============================] - 0s
22.864s loss; 1.614925088 acc; 0.194711538 val_loss; 1.614974635 val_acc; 0.200000000
Epoch 12/120
250/250 [==============================] - 0s
22.761s loss; 1.612979597 acc; 0.206330128 val_loss; 1.614891129 val_acc; 0.200000000



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
	if prev == 'loss;':
		losses.append(float(stokens[i]))
	if prev == 'acc;':
		accs.append(float(stokens[i]))
	if prev == 'val_loss;':
		val_losses.append(float(stokens[i]))



