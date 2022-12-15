# Import libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
import pickle


# clear equation directory for new equation images
files = glob.glob('../image_data/image_output/*')
#files = glob.glob('../image_data/image_output/')
for f in files:
    os.remove(f)


# Load in equation image
#img_path = '../image_data/test_equations/test20.jpeg'
img_path = '../image_data/equation_imput/test.jpeg'
img = cv2.imread(img_path)


## the following code was adapted from "https://www.appsloveworld.com/opencv/100/73/python-split-an-image-based-on-white-space"
## in order to prepare the image for drawing bounding boxes.

# define border color
lower = (0, 80, 110)
upper = (0, 120, 150)

# threshold on border color
mask = cv2.inRange(img, lower, upper)

# set threshold
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

# recolor border to white
img[mask==255] = (255,255,255)

# convert img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find threshold
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 

# apply morphology open
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
morph = 255 - morph

# Find coordinates for bounding boxes
bboxes = []
bboxes_img = img.copy()
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    x,y,w,h = cv2.boundingRect(cntr)
    cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    bboxes.append((x,y,w,h))


# get largest width of bboxes
maxwidth = max(bboxes)[2]

# sort bboxes on x coordinate
def takeFirst(elem):
    return elem[0]

bboxes.sort(key=takeFirst)

# seperate cropped images and write to file
#result = np.full((1,maxwidth+20,3), (255,255,255), dtype=np.uint8)
i = 1
for bbox in bboxes:
    (x,y,w,h) = bbox
    #crop = img[y - 10:y + h + 50, x - 50:x + maxwidth + 80]
    #crop = img[y - 30:y + h + 50, x - 50:x + maxwidth + 30]
    crop = img[y - 40:y + h + 40, x-40:x + maxwidth + 30]
    #result = np.vstack((result, crop))
    cv2.imwrite(f'../image_data/image_output/test_char{str(i)}.jpg', crop)
    i += 1
# save indermediate results
cv2.imwrite('../image_data/processed_images/img_mask.jpg', mask)
cv2.imwrite('../image_data/processed_images/img_border.jpg', img)
cv2.imwrite('../image_data/processed_images/img_thresh.jpg', thresh)
cv2.imwrite('../image_data/processed_images/img_morph.jpg', morph)
cv2.imwrite('../image_data/processed_images/img_bboxes.jpg', bboxes_img)