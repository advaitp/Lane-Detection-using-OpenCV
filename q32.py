import cv2
import numpy as np
from math import sqrt
import argparse

def extractRegion(img) :
	# Create a mask to crop that region
	rows, cols = img.shape[:2]
	bottom_left  = [cols * 0.1, rows * 0.95]
	top_left     = [cols * 0.4, rows * 0.6]
	bottom_right = [cols * 0.9, rows * 0.95]
	top_right    = [cols * 0.6, rows * 0.6]
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	# vertices = np.array([[40,570], [220,330], [600,330], [920,570]])

	# Applying the mask 
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, [255,255,255])
	masked = cv2.bitwise_and(img, mask)

	# Applying threshold
	ret, thresh = cv2.threshold(masked, 175, 255, cv2.THRESH_BINARY)
	return thresh

def drawlines(img, lines, color) :
	for line in lines:
		if line is not None:
			cv2.line(img, line[0], line[1],  color, 4)  
	return img

def getPoints(pimg) :
	h, w = pimg.shape[0], pimg.shape[1]
	gimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
	linesP = cv2.HoughLinesP(gimg, 1, np.pi / 180, 50, 10, 0)
	left, right = [], []

	for line in linesP:
		if line is not None:
			for x1, y1, x2, y2 in line:
				if x1 < int(w/2) :
					left.append([x1, y1, x2, y2])
				else :
					right.append([x1, y1, x2, y2])

	return left, right

def getDist(linesP) :
	distance = []
	for x1, y1, x2, y2 in linesP:
		dist = (x1-x2)**2+(y1-y2)**2
		distance.append(dist)

	if len(distance) == 0 : return 0
	return max(distance)

def drawlines(img, lines, color) :
	for line in lines:
		cv2.line(img, (line[0],line[1]), (line[2], line[3]),  color, 4)  
	return img

if __name__ == "__main__" :
	# img = cv2.imread('Frame6.jpg')
	# cv2.imshow('ROI', new_img)
	# cv2.waitKey(0)

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--VideoPath', default="whiteline.mp4", help='base path where data files exist')
	Args = Parser.parse_args("")
	videoPath = Args.VideoPath
	video = cv2.VideoWriter('Lane_Detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (960,540))
	cap = cv2.VideoCapture(videoPath)

	if(cap.isOpened()== False): 
		print("Error opening video stream or file")

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			framec = frame.copy()
			framec1 = frame.copy()

			pframe1 = extractRegion(framec)
			leftp, rightp = getPoints(pframe1)
			adist1 = getDist(leftp)
			adist2 = getDist(rightp)

			if adist1 > adist2 : 
				frame = drawlines(frame, leftp, [0,255,0])
				frame = drawlines(frame, rightp, [0,0,255])
			else :
				frame = drawlines(frame, rightp, [0,255,0])
				frame = drawlines(frame, leftp, [0,0,255])

			cv2.imshow('Frame', frame)
			video.write(frame)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else: 
			break

	cap.release()
	cv2.destroyAllWindows()
	video.release()  