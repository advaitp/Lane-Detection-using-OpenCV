import cv2
import numpy as np
import argparse

def extractRegion(img) :
	# Create a mask to crop that region
	rows, cols = img.shape[:2]
	bottom_left  = [cols * 0.12, rows * 0.95]
	top_left     = [cols * 0.45, rows * 0.6]
	bottom_right = [cols * 0.9, rows * 0.95]
	top_right    = [cols * 0.6, rows * 0.6]
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	# vertices = np.array([[40,570], [220,330], [600,330], [920,570]])

	# Applying the mask to crop
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, [255, 255, 255])
	masked = cv2.bitwise_and(img, mask)

	# Applying threshold
	ret, thresh = cv2.threshold(masked, 240, 255, cv2.THRESH_BINARY)
	return masked

def createBirdview(img) :
	imgc = img.copy()
	rows, cols = img.shape[:2]
	bottom_leftp  = [int(cols * 0.17), int(rows * 0.94)]
	top_leftp     = [int(cols * 0.41), int(rows * 0.65)]
	bottom_rightp = [int(cols * 0.88), int(rows * 0.94)]
	top_rightp    = [int(cols * 0.61), int(rows * 0.65)]

	# # Get image from bird's view
	rect = np.float32([top_leftp, bottom_leftp, bottom_rightp, top_rightp])
	dst = np.float32([[0, 0], [0, 500], [500,500], [500,0]])
	H = cv2.getPerspectiveTransform(rect, dst)	
	warped = cv2.warpPerspective(img, H, (500, 500), flags=cv2.INTER_LINEAR)

	return warped, H, rect

def stitchImage(img1, img2):
	img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	img1c=img1.copy()

	img1[:,:,0] = mask_inv
	img1[:,:,1] = mask_inv
	img1[:,:,2] = mask_inv
	
	final_mask = cv2.bitwise_and(img1,img1c)
	dst = cv2.add(img2, final_mask)
	return dst

def curveFit(img) : 
	imgc = img.copy()
	h, w = imgc.shape[0], imgc.shape[1]
	ret, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

	gimg = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

	# # # Preprocessing Image
	imgHLS = cv2.cvtColor(thresh, cv2.COLOR_BGR2HLS)
	mask = cv2.inRange(imgHLS[:,:,1], 120, 255)
	res = cv2.bitwise_and(img, img, mask= mask)
	# res = cv2.equalizeHist(res)

	lxpts, lypts, rxpts, rypts = [], [], [], []
	corners = cv2.goodFeaturesToTrack(gimg,1000,0.01,10)
	corners = np.int0(corners)

	for i in corners:
	    x, y = i.ravel()
	    # cv2.circle(imgc, (x,y), 5, (0,255,0), -1)
	    if x > w // 2 :
	    	rxpts.append(x)
	    	rypts.append(y)
	    else :
	    	lxpts.append(x)
	    	lypts.append(y)

	# # #  x = ay**2+by+c
	lfit = np.polyfit(lypts, lxpts, 2)
	rfit = np.polyfit(rypts, rxpts, 2)

	al, ar = lfit[0], rfit[0]
	bl, br = lfit[1], rfit[1]
	cl, cr = lfit[2], rfit[2]
	y = np.arange(h)
	lpoints = al * np.square(y) + bl * y + cl
	rpoints = ar * np.square(y) + br * y + cr
	lxpoints = lpoints.astype(np.uint32)
	rxpoints = rpoints.astype(np.uint32)

	lradius = ((1 + (2*al*y+bl)**2)**1.5) / np.absolute(2*al)
	rradius = ((1 + (2*ar*y+br)**2)**1.5) / np.absolute(2*ar)
	lradius = np.mean(lradius)
	rradius = np.mean(rradius)
	radius = np.mean(lradius+rradius)//2

	# radius = f'Radius of curvature is {np.mean(lradius+rradius)//2}(m)'
	# lradius = f'Left Radius of curvature is {np.mean(lradius)}(m)'
	# rradius = f'Right Radius of curvature is {np.mean(rradius)}(m)'

	lanecenter = (rxpoints-lxpoints)//2 + lxpoints
	imagecenter = w//2
	diff = np.mean(lanecenter-imagecenter)
	
	threshold = 2
	if lradius < 0 or rradius < 0 or radius < 0 or diff > 500: 
		turnP = False
		turn = f'Vehicle is straight'
		move = 'Not found!'
	else : 
		turnP = True
		if diff > threshold and diff < 500:
			move = 'Turn Right'
			turn = f'Vehicle is {abs(diff)}m right of center'
		elif diff < -threshold: 
			move = 'Turn Left'
			turn = f'Vehicle is {abs(diff)}m left of center'
		else : 
			move = 'Go Straight'
			turn = f'Vehicle is straight'

	lpoints = np.stack((lxpoints, y), axis = 1)
	rpoints = np.stack((rxpoints, y), axis = 1)
	centerpoints = np.stack((lanecenter, y), axis = 1)

	# Plot the polylines and area between them
	# points=np.concatenate((lpoints, rpoints), axis = 0)
	points = np.array([lpoints[0], lpoints[-1], rpoints[-1], rpoints[0]])
	# cv2.fillPoly(res, pts = [points], color =(0,0,255))
	res = cv2.polylines(res, [centerpoints], False, (0,0,255), 2)
	res = cv2.polylines(res, [lpoints], False, (0,255,0), 4)
	res = cv2.polylines(res, [rpoints], False, (0,255,0), 4)

	cv2.fillPoly(imgc, pts = [points], color = (180,180,255))
	imgc = cv2.polylines(imgc, [centerpoints], False, (0,0,255), 2)
	imgc = cv2.polylines(imgc, [lpoints], False, (0,255,255), 5)
	imgc = cv2.polylines(imgc, [rpoints], False, (0,255,255), 5)

	return res, imgc, turn, radius, lradius, rradius, move, thresh, turnP

def retrieveImage(wimg, H, img) : 
	Hinv = np.linalg.inv(H)
	h, w = img.shape[0], img.shape[1]
	nimg = cv2.warpPerspective(wimg, Hinv, (w, h), flags=cv2.INTER_LINEAR)
	oimg = stitchImage(img, nimg)
	return oimg

def createInterface(frame, nframe, thresh, pcframe, thresholded, info) :
	display = np.ones((720, 1080, 3), dtype = "uint8")

	resized1 = cv2.resize(nframe, (660,480), interpolation = cv2.INTER_AREA)
	display[:480, :660, :] = resized1

	resized2 = cv2.resize(frame, (210,120), interpolation = cv2.INTER_AREA)
	display[0:120, 660:870, :] = resized2

	resized3 = cv2.resize(thresh, (210,120), interpolation = cv2.INTER_AREA)
	display[0:120, 870:1080, :] = resized3

	resized4 = cv2.resize(thresholded, (210,360), interpolation = cv2.INTER_AREA)
	display[120:480, 660:870, :] = resized4

	resized5 = cv2.resize(pcframe, (210,360), interpolation = cv2.INTER_AREA)
	display[120:480, 870:1080, :] = resized5

	display[480:720, :1080, :] = [227, 195, 203]*np.ones((240, 1080, 3), dtype = "uint8")
	# [turn, radius, lradius, rradius, move, turnP]

	information = '(1): Undistorted image, (2): Detected white and yellow markings, (3): Warped Image, (4): Detected points and curve fitting'
	font = cv2.FONT_HERSHEY_SIMPLEX
	color = [255, 0, 0]
	color1 = [0, 0, 255]
	color2 = [255,255,255]

	display = cv2.putText(display, info[4], (50,50), font,
	                   0.8, color1, 2, cv2.LINE_AA)
	display = cv2.putText(display, information, (10,500), font,
	                   0.5, color, 1, cv2.LINE_AA)
	display = cv2.putText(display, 'Right Curvature: '+str(info[3]), (10,530), font,
	                   0.5, color, 1, cv2.LINE_AA)
	display = cv2.putText(display, 'Left Curvature: '+str(info[2]), (400,530), font,
	                   0.5, color, 1, cv2.LINE_AA)

	if info[5] : 
		display = cv2.putText(display, 'Average Curvature: '+str(info[1]), (10,580), font,
	                   0.5, color, 1, cv2.LINE_AA)
	else :
		display = cv2.putText(display, 'Curvature not found!', (10,580), font,
	                   0.5, color, 1, cv2.LINE_AA)

	display = cv2.putText(display, '(1)', (670,20), font,
                   0.5, color, 1, cv2.LINE_AA)
	display = cv2.putText(display, '(2)', (880,20), font,
	                   0.5, color, 1, cv2.LINE_AA)
	display = cv2.putText(display, '(3)', (670,170), font,
	                   0.5, color, 1, cv2.LINE_AA)
	display = cv2.putText(display, '(4)', (880,170), font,
	                   0.5, color, 1, cv2.LINE_AA)

	return display

if __name__ == "__main__" :
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--VideoPath', default="challenge.mp4", help='base path where data files exist')
	Args = Parser.parse_args("")
	videoPath = Args.VideoPath
	video = cv2.VideoWriter('Turn_Detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1080, 720))
	cap = cv2.VideoCapture(videoPath)

	cap = cv2.VideoCapture(videoPath)
	if(cap.isOpened()== False): 
		print("Error opening video stream or file")

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			information = []
			framec = frame.copy()
			framec1 = frame.copy()
			framec2 = frame.copy()

			pframe = extractRegion(framec)
			ret, thresh = cv2.threshold(pframe, 230, 255, cv2.THRESH_BINARY)
	
			bframe, H, crop = createBirdview(pframe) 
			pcframe, cframe, turn, radius, lradius, rradius, move, thresholded, turnP = curveFit(bframe)
			nframe = retrieveImage(cframe, H, frame)
			information.extend([turn, radius, lradius, rradius, move, turnP])

			interface = createInterface(framec2, nframe, thresh, pcframe, thresholded, information)
			cv2.imshow('Frame', interface)
			video.write(interface)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else: 
			break

	cap.release()
	cv2.destroyAllWindows()
	video.release()  