import cv2
import numpy as np
from scipy.ndimage import label
camera = cv2.VideoCapture(0) #Insert the camera number here
ret = camera.set(3,1920)
ret = camera.set(4,1080)


def main():
	threshold_value=100
	SW = 0
	EW = 760
	SH = 0
	EH = 800
	key = 0

	MainMenu()
	#Grayscale()
	#image = GrabImage()
	#threshold_value = BlackWhite(threshold_value)
	#cv2.imshow('image',image)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#threshold_value = BlackWhite(threshold_value)

	#SW,EW,SH,EH = GenerateImage(SW,EW,SH,EH)
	#cv2.destroyAllWindows()
	#Watershed()



def display(windowName, image):
  cv2.namedWindow(windowName, 1)
  cv2.imshow(windowName, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows() 



def MainMenu():
	while(1):

		Grayscale()
		#threshold_value=BlackWhite()



def Grayscale():
	while(1):
		#change to change image size
		SW = 300 #starting width
		EW = 1200#ending width
		SH = 100#starting hight
		EH = 800#ending hight
		im_uncropped = camera.read()[1] # read from webcam
		
		cv2.rectangle(im_uncropped, (SW,SH), (EW,EH), (255,255,255))
		im_gray = cv2.cvtColor(im_uncropped,cv2.COLOR_BGR2GRAY) #convert to BW
		#im_gray = im_uncropped [50:700,50:400]
		cv2.putText(im_gray, "main menu", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,10,0))
		cv2.putText(im_gray, "Press \"Space\" to take picture", (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,10,0))
		cv2.putText(im_gray, "Press \"1\" perform Watershed on last picture", (0,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,10,0))
		cv2.putText(im_gray, "Press \"2\" perform inverse_Watershed on last picture", (0,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,10,0))
		cv2.putText(im_gray, "Press \"3\" perform skeletonise on last picture", (0,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,10,0))
		cv2.putText(im_gray, "Press \"4\" perform manual threshold watershed on last picture", (0,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,10,0))
		cv2.putText(im_gray, "Press \"5\" to change manual settings ", (0,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,10,0))
		cv2.putText(im_gray, "Press ESC to exit", (0,190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,10,0))

		cv2.imshow('BW image2',im_gray) # Display to window
		key=cv2.waitKey(5)
		if key==27: #close window on escape
			status=0
			exit()
		if key==32:
			cv2.destroyAllWindows()
			im_uncropped = camera.read()[1]
			im = im_uncropped [SH:EH,SW:EW]
			cv2.imwrite("temp.jpg",im)
			#return
		if key==49:
			Watershed()
			return
		if key==50:
			Watershed_inverse()
			return
		if key==51:
			skeletonise()
			return
		if key==52:
			BlackWhite()
			return
		if key==53:
			return



		if key==65364:
			threshold_value=threshold_value+5
		if key==65362:
			threshold_value=threshold_value-5


	return

def GrabImage():
	for i in range(0,5):
		im = camera.read()[1]
		cv2.waitKey(1)
	return (im)

def BlackWhite(threshold_value=100):
	while(1):
		im_live = camera.read()[1] # read from webcam
		#im_live = cv2.medianBlur(im_live,5)
		im_gray_live = cv2.cvtColor(im_live,cv2.COLOR_BGR2GRAY)
		ret,im_BW_live = cv2.threshold(im_gray_live,threshold_value,255,0) #convert to BW
		cv2.imshow('BW image2',im_BW_live) # Display to window
		#cv2.imshow('adaptiveThreshold',th3)
		key=cv2.waitKey(5)
		if key==32:
			cv2.destroyAllWindows()
			#im = camera.read()[1]
			cv2.imwrite("temp_edited.jpg",im_BW_live)
			return
		if key==65364:
			threshold_value=threshold_value+5
		if key==65362:
			threshold_value=threshold_value-5
	return (threshold_value)


def Watershed_inverse():
		# Read image
	img = cv2.imread('temp.jpg')
	# Convert to grayscale image
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	display('gray', gray)
	# Convert to binary image
	ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


	# noise removal
	# This part of the function removes noise
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	sure_bg = cv2.dilate(opening,kernel,iterations=3)
	display('Sure Background', sure_bg)

	dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
	display('Sure Foreground', sure_fg)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)
	display('unknown area', unknown)


	contours, hierarchy = cv2.findContours (sure_fg,cv2.cv.CV_RETR_EXTERNAL,cv2.cv.CV_CHAIN_APPROX_SIMPLE)


	seed_number = len(contours)
	print ("number of seeds= ",seed_number)

	ret,sure_bg = cv2.threshold(sure_bg,0,255,cv2.THRESH_BINARY_INV)
	lbl, ncc = label(sure_fg)
	lbl = lbl * (255/ncc)
	lbl[sure_bg == 255] = 255
	lbl = lbl.astype(np.int32)
	cv2.watershed(img, lbl)

	lbl[lbl == -1] = 0
	lbl = lbl.astype(np.uint8)
	result = 255 - lbl
	#display("Final image", result )

	result[result != 255] = 0
	result = cv2.dilate(result, None)
	img[result == 255] = (0, 0, 255)
	#display("Final image", img )

	cv2.putText(img, "number of seeds= " + str(seed_number), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
	#cv2.drawContours(img, contours, -1, (0,255,255), 3)

	display("Final image", img )
	return


def Watershed():
		# Read image
	thresh = cv2.imread('temp.jpg')
	# Convert to grayscale image
	display('gray', gray)
	# Convert to binary image


	# noise removal
	# to remove any small white noises use morphological opening
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	sure_bg = cv2.dilate(opening,kernel,iterations=3)
	display('Sure Background', sure_bg)

	dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
	display('Sure Foreground', sure_fg)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)
	display('unknown area', unknown)


	contours, hierarchy = cv2.findContours (sure_fg,cv2.cv.CV_RETR_EXTERNAL,cv2.cv.CV_CHAIN_APPROX_SIMPLE)

	ret,sure_bg = cv2.threshold(sure_bg,0,255,cv2.THRESH_BINARY_INV)
	lbl, ncc = label(sure_fg)
	lbl = lbl * (255/ncc)
	lbl[sure_bg == 255] = 255
	lbl = lbl.astype(np.int32)
	cv2.watershed(img, lbl)

	lbl[lbl == -1] = 0
	lbl = lbl.astype(np.uint8)
	result = 255 - lbl
	#display("Final image", result )

	result[result != 255] = 0
	result = cv2.dilate(result, None)
	img[result == 255] = (0, 0, 255)
	seed_number = len(contours)
	print ("number of seeds= ",seed_number)


	cv2.putText(img, "number of seeds= " + str(seed_number), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
	#cv2.drawContours(img, contours, -1, (0,255,255), 3)

	display("Final image", img )
	return

def Watershed_manual():
		# Read image
	img = cv2.imread('temp_edited.jpg')
	# Convert to grayscale image
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	display('gray', gray)
	# Convert to binary image
	ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


	# noise removal
	# to remove any small white noises use morphological opening
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	sure_bg = cv2.dilate(opening,kernel,iterations=3)
	display('Sure Background', sure_bg)

	dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
	display('Sure Foreground', sure_fg)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)
	display('unknown area', unknown)


	contours, hierarchy = cv2.findContours (sure_fg,cv2.cv.CV_RETR_EXTERNAL,cv2.cv.CV_CHAIN_APPROX_SIMPLE)

	ret,sure_bg = cv2.threshold(sure_bg,0,255,cv2.THRESH_BINARY_INV)
	lbl, ncc = label(sure_fg)
	lbl = lbl * (255/ncc)
	lbl[sure_bg == 255] = 255
	lbl = lbl.astype(np.int32)
	cv2.watershed(img, lbl)

	lbl[lbl == -1] = 0
	lbl = lbl.astype(np.uint8)
	result = 255 - lbl
	#display("Final image", result )

	result[result != 255] = 0
	result = cv2.dilate(result, None)
	img[result == 255] = (0, 0, 255)
	seed_number = len(contours)
	print ("number of seeds= ",seed_number)


	cv2.putText(img, "number of seeds= " + str(seed_number), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
	#cv2.drawContours(img, contours, -1, (0,255,255), 3)

	display("Final image", img )
	return

def GenerateImage(Starting_width,Ending_width,Starting_hight,Ending_hight):

    filename = "temp.jpg"
    im_uncropped = cv2.imread(filename)
    im = im_uncropped [Starting_width:Ending_width,Starting_hight:Ending_hight]



    #im = cv2.medianBlur(im,3)
    new = im.copy()

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,100,255,0) ## determinate objects
    kernel= np.ones((3,3),np.uint8)
    for i in range(0,10):
    	thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    	thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
    thresh = cv2.erode(thresh,kernel,iterations=3)

    #cv2.imshow("Image/BW",thresh)

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imshow("original image",im)
    #print "PULSE TO ANY KEY TO CONTINUE"
    cv2.drawContours( im,contours,-1,(0,255,0),5)  
    cv2.imshow("partitioned window", im)

    for h,cnt in enumerate(contours):
        if (len(cnt)>5):
            Carea=cv2.contourArea(cnt)
            #print(Carea)
            if(Carea>50)&(Carea<1500):
            	
                mask = np.zeros(imgray.shape,np.uint8)
                color = cv2.mean(im,mask = mask)
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(im,ellipse,(0,255,255),2)
    key=cv2.waitKey(5)
    if key==32:
        cv2.destroyAllWindows()
        exit(0)
    if key==65364:
        Starting_hight=Starting_hight+5
        print"Starting_hight = ",Starting_hight

    if key==65362:
        Starting_hight=Starting_hight-5
        print"Starting_hight = ",Starting_hight
    cv2.imshow("Image",im)
    if key==65363:
        Ending_hight=Ending_hight+5
        print"Ending_hight = ",Ending_hight

    if key==65361:
        Ending_hight=Ending_hight-5
        print"Ending_hight = ",Ending_hight
    cv2.imshow("Image",im)
    return (Starting_width,Ending_width,Starting_hight,Ending_hight)

def skeletonise():

	img = cv2.imread('temp.jpg')
	#img = im_uncropped [SW:EW,SH:EH]

	 

	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	kernel= np.ones((3,3),np.uint8)
	img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
	img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
	cv2.imshow("image",img)
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False
	 
	while( not done):
	    eroded = cv2.erode(img,element)
	    temp = cv2.dilate(eroded,element)
	    temp = cv2.subtract(img,temp)
	    skel = cv2.bitwise_or(skel,temp)
	    img = eroded.copy()
	 
	    zeros = size - cv2.countNonZero(img)
	    if zeros==size:
	        done = True
	 

	#for i in range(0,1):
	#	kernel= np.ones((2,2),np.uint8)
	#	skel = cv2.morphologyEx(skel,cv2.MORPH_CLOSE,kernel)
	#	skel = cv2.morphologyEx(skel,cv2.MORPH_OPEN,kernel)
	cv2.imshow("skel",skel)



	cv2.waitKey(0)
	cv2.destroyAllWindows()




main()
