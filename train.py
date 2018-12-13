import sys
sys.setrecursionlimit(10000)
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Lambda, TimeDistributed, Conv2D, BatchNormalization, Activation
from keras.layers import concatenate, Concatenate, AveragePooling2D, UpSampling2D, Flatten, MaxPool2D
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tqdm
import pickle as pkl
from keras.callbacks import TensorBoard
from time import time
from keras.activations import relu
from keras.models import Model
K.set_image_dim_ordering('th')
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import glob
from dataloader import generateSampleFace
from model import *
from random import shuffle


datasets = ['AFW', 'HELEN', 'IBUG', 'LFPW']
landmark_path = '/ssd_scratch/cvit/abhishek/300W_LP/landmarks/'

pts_list=[]
for i in datasets:
	pts_temp = glob.glob('/ssd_scratch/cvit/abhishek/300W_LP/landmarks/'+i+'/*.t7')
	pts_list+=pts_temp

shuffle(pts_list)
# len_iter = -1
# x_total=np.zeros((len(pts_list[:len_iter]),256,256,3))
# y_total=np.zeros((len(pts_list[:len_iter]),20,64,64))



# for i in tqdm.tqdm(range(0,len(pts_list[:len_iter]))):
# 	lm_file = pts_list[i]
# 	img_temp = lm_file.split('/')
# 	img_temp.pop(-3)
# 	img_temp[-1] = img_temp[-1].split('_pts.t7')[0]+'.jpg'
# 	img_file = ('/').join(img_temp)
# 	x_temp, y_temp, _, _, _ = generateSampleFace(img_file,lm_file)
# 	x_total[i] = x_temp
# 	y_total[i] = y_temp



def my_generator(pts_list):
	while 1:
		batchSize = 16
		for i0 in range(0,len(pts_list),batchSize):
			x_batch=np.zeros((batchSize,3,256,256))
			y_batch=np.zeros((batchSize,20,64,64))
			# numi1=0
			for i1 in range(0,batchSize):
				i2 = i0+i1
				# print(i2)
				lm_file = pts_list[i2]
				img_temp = lm_file.split('/')
				img_temp.pop(-3)
				img_temp[-1] = img_temp[-1].split('_pts.t7')[0]+'.jpg'
				img_file = ('/').join(img_temp)
				x_temp, y_temp, _, _, _ = generateSampleFace(img_file,lm_file)
				# print(x_temp.shape,x_batch.shape)
				x_batch[i1] = x_temp.transpose(2,0,1)
				y_batch[i1] = y_temp
			yield (x_batch, [y_batch,y_batch,y_batch,y_batch])

# [y_batch,y_batch,y_batch,y_batch]

x = Input(shape=(3, 256, 256))
# cb = ResNetDepth(x)
cb = FAN(x,4)
model = Model(input=x,outputs=cb)
model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
model.summary()


spe_train = int(len(pts_list[:-1000])/16)
spe_val = int(len(pts_list[-1000:])/16)

for i in range(0,10):
	try:
		history_of_model=model.fit_generator(my_generator(pts_list[:-1000]), steps_per_epoch=spe_train, validation_data=my_generator(pts_list[-1000:]),nb_epoch=300, verbose=1, nb_worker=1,validation_steps=spe_val, shuffle=True)
	except:
		model.save_weights('saved_models/'+str(i)+'.h5')
		print(i)
