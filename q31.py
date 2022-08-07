import numpy as np 
import cv2
import matplotlib.pyplot as plt
import os
import argparse

def contrast_adjust(hist) :
	clip_limit = 40
	indexes = []
	total = 0
	
	for i in range(256) :
		if hist[i] >= clip_limit :
			total += (hist[i]-clip_limit)
			hist[i] = clip_limit
		else :
			indexes.append(i)

	increment = int(total / len(indexes))

	for i in indexes :
		hist[i] += increment

	out_hist = np.clip(hist, a_min = 0, a_max = clip_limit)
	return hist

def createhist(image, flag) :
	hist = np.zeros(256)
	m, n = np.unique(image.flatten(), return_counts = True, axis=0)
	hist[m] = n
	if flag : 
		hist = contrast_adjust(hist)
	return hist

def equalizer(img, flag) :
	hist = createhist(img, flag)
	num_pixels = np.sum(hist)
	hist = hist/num_pixels
	cum_hist = np.cumsum(hist)
	cdf = np.floor(255 * cum_hist).astype(np.uint8)

	img_list = list(img.flatten())
	
	eq_img_list = [cdf[p] for p in img_list]

	eq_img = np.array(eq_img_list).astype('float32').reshape(img.shape)

	return eq_img

def adaptive_hist_equalization(img) :
	h, w, ch = img.shape 
	# tile_size = 8
	# for r in range(0, h-tile_size, tile_size) :
	# 	for c in range(0, w-tile_size, tile_size) :
	# 		img[r:r+tile_size, c:c+tile_size] = hist_equalization(img[r:r+tile_size, c:c+tile_size], 0)

	num_tiles = 8 
	stepx = h//num_tiles
	stepy = w//num_tiles 

	for i in range(0,num_tiles-1) :
		for j in range(0,num_tiles-1) : 
			img[i*stepx:(i+1)*stepx, j*stepy:(j+1)*stepy] = hist_equalization(img[i*stepx:(i+1)*stepx, j*stepy:(j+1)*stepy], 1)

	return img

def visualise_hist(img) :
	lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab_img)
	plt.hist(l.flat, bins=256)
	plt.show()

def hist_equalization(img, flag=0) : 
	img[:, :, 0] = equalizer(img[:, :, 0], flag)
	img[:, :, 1] = equalizer(img[:, :, 1], flag)
	img[:, :, 2] = equalizer(img[:, :, 2], flag)

	return img

if __name__ == "__main__" :

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--DirPath', default="/adaptive_hist_data", help='base path where data files exist')
	Args = Parser.parse_args("")
	DirPath = Args.DirPath
	video = cv2.VideoWriter('Histogram_Equalization.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (1224,1110))
	
	imageDir = os.getcwd() + DirPath
	for imgP in os.listdir(imageDir) :
		imgP = os.path.join(imageDir, imgP)
		img = cv2.imread(imgP)
		imgc = img.copy()
		imgc1 = img.copy()
		# visualise_hist(img)
		img_e = hist_equalization(imgc)
		# visualise_hist(img_e)

		img_a = adaptive_hist_equalization(imgc1)
		# visualise_hist(img_a)

		# cv2.imshow('Color input image', img)
		# cv2.imshow('Histogram equalized', img_e)
		# cv2.imshow('Adaptive Histogram equalized', img_a)
		img_o = np.concatenate((img, img_a, img_e))

		font = cv2.FONT_HERSHEY_SIMPLEX
		color = [0, 0, 255]
		img_o = cv2.putText(img_o, 'Normal Video', (10,50), font, 0.5, color, 1, cv2.LINE_AA)
		img_o = cv2.putText(img_o, 'Adaptive Histogram Equalization', (10,450), font, 0.5, color, 1, cv2.LINE_AA)
		img_o = cv2.putText(img_o, 'Histogram Equalization', (10,850), font, 0.5, color, 1, cv2.LINE_AA)
		video.write(img_o)
		cv2.imshow('Output equalized', img_o)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	video.release()  
	
