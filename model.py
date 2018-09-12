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
from tqdm import tqdm
import pickle as pkl
from keras.callbacks import TensorBoard
from time import time
from keras.activations import relu
from keras.models import Model
K.set_image_dim_ordering('th')
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def conv3x3(in_planes, out_planes, strd=1, padding='same', bias=False):
	"3x3 convolution with padding"
	return Conv2D(out_planes, kernel_size=3, strides=(1, 1), padding=padding, use_bias=bias)


def ConvBlock(input, in_planes, out_planes):
	residual = input
	in_planes = input.shape[1]
	# print(input.shape[1])
	bn1 = BatchNormalization(axis=1,momentum=0.1)(input)
	act1 = Activation('relu')(bn1)
	conv1 = conv3x3(in_planes, int(out_planes / 2))(act1)
	bn2 = BatchNormalization(axis=1,momentum=0.1)(conv1)
	act2 = Activation('relu')(bn2)
	conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))(act2)
	bn3 = BatchNormalization(axis=1,momentum=0.1)(conv2)
	act3 = Activation('relu')(bn3)
	conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))(act3)
	merge1 = Concatenate(axis=1)([conv1,conv2,conv3])
	# merge1 = merge([conv1,conv2,conv3], mode = 'concat', concat_axis = -1)
	# out = Model(input=bn1, output=merge1)
	if in_planes != out_planes:
		bn4 = BatchNormalization(axis=1,momentum=0.1)(residual)
		act4 = Activation('relu')(bn4)
		conv4 = Conv2D(out_planes, kernel_size=1, strides=1, padding='same', use_bias=False)(act4)
		# out = Model(input=bn1, output=conv4)
		residual = conv4
	out = merge([merge1,residual], mode = 'sum', concat_axis = 1)
	return out



def Bottleneck(input, inplanes, planes, strides=1, downsample=None):
	residual = input
	conv1 = Conv2D(planes, kernel_size=1, use_bias=False)(input)
	bn1 = BatchNormalization(axis=1,momentum=0.1)(conv1)
	act1 = Activation('relu')(bn1)
	conv2 = Conv2D(planes, kernel_size=3, strides=strides, padding='same', use_bias=False)(act1)
	bn2 = BatchNormalization(axis=1,momentum=0.1)(conv2)
	act2 = Activation('relu')(bn2)
	conv3 = Conv2D(planes * 4, kernel_size=1, use_bias=False)(act2)
	bn3 = BatchNormalization(axis=1,momentum=0.1)(conv3)
	if downsample is not None:
		# residual = downsample
		residual = Conv2D(downsample[0], kernel_size=1, strides=downsample[1], use_bias=False)(input)
		residual = BatchNormalization(axis=1,momentum=0.1)(residual)
	out = merge([bn3,residual], mode = 'sum', concat_axis = 1)
	out = Activation('relu')(out)
	return out









def HourGlass(input, num_modules, depth, num_features):
	# level =  depth
	def forward(level, inp):
	# for level in range(depth,0,-1):
		up1 = inp
		# exec('up1 = b1_' + str(level)+'(up1)')
		up1 = ConvBlock(up1, 256, 256)
		low1 = AveragePooling2D(2, strides=2)(inp)
		# exec('low1 = b2_' + str(level)+'(low1)')
		low1 = ConvBlock(low1, 256, 256)
		if  level > 1:
			low2 = forward(level - 1, low1)
		else:
			low2 = low1
			# exec('low2 = b2_plus_' + str(level)+'(low2)')
			low2 = ConvBlock(low2, 256, 256)
		low3 = low2
		# exec('low3 = b3_' + str(level)+'(low3)')
		low3 = ConvBlock(low3, 256, 256)
		# up2 = F.upsample(low3, scale_factor=2, mode='nearest')
		up2 = UpSampling2D(size=(2, 2))(low3)
		# return up1+up2
		return merge([up1,up2], mode = 'sum', concat_axis = 1)
	return forward(depth,input)










# def HourGlass(input, num_modules, depth, num_features):
# 	# level =  depth
# 	for level in range(depth,0,-1):
# 		exec('b1_' + str(level)+' = ConvBlock(256, 256)')
# 		exec('b2_' + str(level)+' = ConvBlock(256, 256)')
# 		if level ==1:
# 			exec('b2_plus_' + str(level)+' = ConvBlock(256, 256)')
# 		exec('b3_' + str(level)+' = ConvBlock(256, 256)')
# 		inp = input
# 	def forward(level, inp):
# 	# for level in range(depth,0,-1):
# 		up1 = inp
# 		exec('up1 = b1_' + str(level)+'(up1)')
# 		low1 = AveragePooling2D(2, strides=2)(inp)
# 		exec('low1 = b2_' + str(level)+'(low1)')
# 		if  level > 1:
# 			low2 = forward(level - 1, low1)
# 		else:
# 			low2 = low1
# 			exec('low2 = b2_plus_' + str(level)+'(low2)')
# 			low3 = low2
# 		exec('low3 = b3_' + str(level)+'(low3)')
# 		# up2 = F.upsample(low3, scale_factor=2, mode='nearest')
# 		up2 = UpSampling2D(size=(2, 2))(low3)
# 		return up1+up2
# 	return forward(depth,input)





def FAN(input, num_modules=1):
	conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same')(input)
	bn1 = BatchNormalization(axis=1,momentum=0.1)(conv1)
	act1 = Activation('relu')(bn1)

	conv2 = ConvBlock(act1,64, 128)
	avgP1 = AveragePooling2D( 2, strides=2)(conv2)

	conv3 = ConvBlock(avgP1,128, 128)
	conv4 = ConvBlock(conv3,128, 256)

	previous = conv4
	outputs = []

	for i in range(num_modules):
		hg = HourGlass(previous,1, 4, 256)
		ll = hg
		ll = ConvBlock(ll,256, 256)
		ll = Conv2D(256, kernel_size=1, strides=1, padding='same')(ll)#valid
		bn_temp = BatchNormalization(axis=1,momentum=0.1)(ll)
		act_temp = Activation('relu')(bn_temp)

		tmp_out = Conv2D(68, kernel_size=1, strides=1, padding='same')(act_temp)#valid
		outputs.append(tmp_out)

		if i < num_modules - 1:
			ll = Conv2D(256, kernel_size=1, strides=1, padding='same')(act_temp)#valid
			tmp_out_ = Conv2D(256, kernel_size=1, strides=1, padding='same')(tmp_out)#valid
			# previous = previous + ll + tmp_out_
			previous = merge([previous,ll,tmp_out_], mode = 'sum', concat_axis = 1)
	return outputs




def ResNetDepth(input, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=68):
	inplanes = 64
	def make_layer(input, block, planes, blocks, strides=1, inplanes=inplanes):
		downsample = None
		if strides != 1 or inplanes != planes * 4:
			downsample = [planes*4, strides]
		ll = block(input, inplanes, planes, strides, downsample)
		inplanes = planes * 4
		for i in range(1, blocks):
			ll = block(ll, inplanes, planes)
		return ll
	conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same',use_bias=False)(input)
	bn1 = BatchNormalization(axis=1,momentum=0.1)(conv1)
	act1 = Activation('relu')(bn1)
	maxP1 = MaxPool2D(pool_size=3, strides=2, padding='valid')(act1)
	layer1 = make_layer(maxP1, block, 64, layers[0])
	layer2 = make_layer(layer1, block, 128, layers[1], strides=2)
	layer3 = make_layer(layer2, block, 256, layers[2], strides=2)
	layer4 = make_layer(layer3, block, 512, layers[3], strides=2)
	avgP1 = AveragePooling2D(7)(layer4)
	flat1 = Flatten()(avgP1)
	fc = Dense(num_classes)(flat1)

	# def make_layer(input, block, planes, blocks, strides=1):
	# 	downsample = None
	# 	if strides != 1 or inplanes != planes * 4:
	# 		downsample = [planes*4, strides]
	# 	ll = block(input, inplanes, planes, strides, downsample)
	# 	inplanes = planes * 4
	# 	for i in range(1, blocks):
	# 		ll = block(ll, inplanes, planes)
	# 	return ll
	return fc


#######################################################################
print('creating_network\n')
x = Input(shape=(3, 64, 64))
# cb = ResNetDepth(x)
cb = FAN(x,1)
model = Model(input=x,outputs=cb)
model.summary()
# print('saving_network_plot')
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')