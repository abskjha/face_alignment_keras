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
import dlib


from utils import *
from dataloader import *
import cv2


datasets = ['AFW', 'HELEN', 'IBUG', 'LFPW']
landmark_path = '/ssd_scratch/cvit/abhishek/300W_dataset/300W/01_Indoor/'

img_list = glob.glob(landmark_path+'*.png')

x_total=np.zeros((len(img_list),256,256,3))



# face_detector = dlib.cnn_face_detection_model_v1('/home/abhishek/lip_landmark_detection/face_alignment_keras/includes/mmod_human_face_detector.dat')
face_detector = dlib.get_frontal_face_detector()



use_cnn_face_detector = False #True



def detect_faces(image):
	return face_detector(image, 1)





# for i in tqdm.tqdm(range(0,len(pts_list[:len_iter]))):
# 	lm_file = pts_list[i]
# 	img_temp = lm_file.split('/')
# 	img_temp.pop(-3)
# 	img_temp[-1] = img_temp[-1].split('_pts.t7')[0]+'.jpg'
# 	img_file = ('/').join(img_temp)
# 	x_temp, y_temp, _, _, _ = generateSampleFace(img_file,lm_file)
# 	x_total[i] = x_temp
# 	y_total[i] = y_temp





x = Input(shape=(3, 256, 256))
# cb = ResNetDepth(x)
cb = FAN(x,4)
model = Model(input=x,outputs=cb)
# model.load_weights('saved_models/13th_sept_2018_2_weights.h5')
model.load_weights('saved_models/9.h5')
model.summary()

def get_landmarks(input_image, all_faces=False):
	if input_image is not 0:
		if isinstance(input_image, str):
			try:
				# print(input_image)
				image = cv2.imread(input_image)
			except IOError:
				print("error opening file :: ", input_image)
				return None
		else:
			image = input_image

		detected_faces = detect_faces(image)
		if len(detected_faces) > 0:
			landmarks = []
			for i, d in enumerate(detected_faces):
				if i > 0 and not all_faces:
					break
				if use_cnn_face_detector:
					d = d.rect

				center = np.array(
					[d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
					 (d.bottom() - d.top()) / 2.0])
				center[1] = center[1] - (d.bottom() - d.top()) * 0.12
				scale = (d.right() - d.left() +
						 d.bottom() - d.top()) / 195.0
				# print(center,scale)
				inp = crop(image, center, scale)
				inp = np.expand_dims((np.array(inp.transpose(
						(2, 0, 1))).astype('float')),axis=0)
				inp = (inp*1.0)/255
				inp = inp - 0.5

				out = model.predict(inp)[-1]
				# cv2.imwrite('wa.jpg',(out.sum(axis=0)*255).astype('uint8'))
				# if self.flip_input:
				# 	out += flip(self.face_alignemnt_net(flip(inp))
				# 				[-1].data.cpu(), is_label=True)

				pts, pts_img = get_preds_fromhm(out, center, scale)
				# print(pts,pts_img)
				pts, pts_img = np.array(pts).reshape(20, 2) * 4, np.array(pts_img).reshape(20, 2)

				# if self.landmarks_type == LandmarksType._3D:
				# 	heatmaps = np.zeros((68, 256, 256))
				# 	for i in range(68):
				# 		if pts[i, 0] > 0:
				# 			heatmaps[i] = draw_gaussian(
				# 				heatmaps[i], pts[i], 2)
				# 	heatmaps = torch.from_numpy(
				# 		heatmaps).view(1, 68, 256, 256).float()
				# 	if self.enable_cuda:
				# 		heatmaps = heatmaps.cuda()
				# 	depth_pred = self.depth_prediciton_net(
				# 		torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
				# 	pts_img = torch.cat(
				# 		(pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)
				landmarks.append(pts_img)
		else:
			print("Warning: No faces were detected.")
			return None

		return landmarks



#######################################################################

################    Comparision between pytorch fan and keras lip-fan

#######################################################################
# import face_alignment
# from skimage import io
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=True)

# diff_fa_faKeras=np.zeros((10,2))
# for ik in range(10,20):
# 	aa = get_landmarks(img_list[ik])
# 	aa = aa[0]
# 	bb = open(img_list[ik][:-4]+'.pts').readlines()

# 	cc = bb[-21:-1]

# 	dd = np.zeros((20,2))

# 	for n1,i in enumerate(cc):                    
# 		dd[n1,0]=float(i[:-1].split(' ')[0])
# 		dd[n1,1]=float(i[:-1].split(' ')[1])
# 	dd_aa = np.sqrt(((dd-aa)**2).mean())
# 	diff_fa_faKeras[ik-10,0]= dd_aa

# 	i = 10
	
	
# 	inpImg = io.imread(img_list[ik])
# 	ee = fa.get_landmarks(inpImg)
# 	ff = ee[0][-20:]
# 	dd_ff = np.sqrt(((dd-ff)**2).mean())
# 	diff_fa_faKeras[ik-10,1]= dd_ff

# print(diff_fa_faKeras)