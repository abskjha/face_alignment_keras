import numpy as np
from utils import *
from torch.utils.serialization import load_lua
from skimage import io
import cv2

nParts = 20
	
def generateSampleFace(image_filename, pts_filename):
	main_pts = load_lua(pts_filename)
	pts = main_pts[1].numpy()
	c = np.array([int(450/2),int((450/2)+50)])
	s = 1.8
	img = cv2.imread(image_filename)
	inp = crop(img, c, s)
	inp = (inp*1.0)/255
	inp = inp - 0.5
	out = np.zeros((nParts, 64, 64))
	for i in range(0,nParts):
		if pts[i][1]>0:
			out[i] = draw_gaussian(out[i],transform(np.add(pts[i],0),c,s,64),1)
	return inp, out, pts, c, s

aa,bb,pts,c,s = generateSampleFace('/ssd_scratch/cvit/abhishek/300W_LP/AFW/AFW_1051618982_1_0.jpg',
	'/ssd_scratch/cvit/abhishek/landmarks/AFW/AFW_1051618982_1_0_pts.t7')
xx,yy=get_preds_fromhm(np.expand_dims(bb,axis=0), center=c, scale=s)

diff_yy_pts=pts-yy