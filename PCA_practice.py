

import cv2
import numpy as np
from scipy.ndimage import label
import sys


def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255/ncc)
    # Completing the markers now. 
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl

# To display a single image in a window
# Window is destroyed on pressing any key
def display(windowName, image):
  cv2.namedWindow(windowName, 1)
  cv2.imshow(windowName, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

"""
img = cv2.imread('temp.jpg')
# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
_, img_bin = cv2.threshold(img_gray, 0, 255,
        cv2.THRESH_OTSU)
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=int))

display("image",img_bin)
result = segment_on_dt(img, img_bin)
display("image",result)

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
display("image",image)
"""

"""
# Read image
img = cv2.imread('temp.jpg')
# Convert to grayscale image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
display('gray', gray)
# Convert to binary image
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


fg = cv2.erode(thresh,None,iterations = 2)

bgt = cv2.dilate(thresh,None,iterations = 3)
ret,bg = cv2.threshold(bgt,1,128,1)
display('fg', fg)
display('bg', bg)
marker = cv2.add(fg,bg)



marker32 = np.int32(marker)

cv2.watershed(img,marker32)
m = cv2.convertScaleAbs(marker32)

ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
res = cv2.bitwise_and(img,img,mask = thresh)

display("image",res)

"""
# Read image
img = cv2.imread('temp.jpg')
# Convert to grayscale image
new = img.copy()
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


ret,sure_bg = cv2.threshold(sure_bg,0,255,cv2.THRESH_BINARY_INV)
lbl, ncc = label(sure_fg)
lbl = lbl * (255/ncc)
lbl[sure_bg == 255] = 255
lbl = lbl.astype(np.int32)
cv2.watershed(new, lbl)

lbl[lbl == -1] = 0
lbl = lbl.astype(np.uint8)
result = 255 - lbl
display("Final image", result )

result[result != 255] = 0
result = cv2.dilate(result, None)
new[result == 255] = (0, 0, 255)
display("Final image", new )

"""
contours, hierarchy = cv2.findContours (sure_fg,cv2.cv.CV_RETR_EXTERNAL,cv2.cv.CV_CHAIN_APPROX_SIMPLE)


seed_number = len(contours)
print ("number of seeds= ",seed_number)


cv2.putText(img, "number of seeds= " + str(seed_number), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
cv2.drawContours(img, contours, -1, (0,255,255), 3)

display("Final image", img )"""

"""
cnt = contours[4]
cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
for h,cnt in enumerate(contours):
    mask = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    mean = cv2.mean(im,mask = mask)
"""

#display('image',img)
"""
border = -sure_bg

lbl, ncc = label(sure_fg)

lbl = lbl * (255/ncc)
# Completing the markers now. 
lbl[border == 255] = 255
lbl = lbl.astype(np.int32)


markers = cv2.watershed(img, lbl)
#display('image',img)
#img[markers == -1] = [255,0,0]

lbl[lbl == -1] = 0
lbl = lbl.astype(np.uint8)
#display('image',lbl)
img = 255 - lbl
"""