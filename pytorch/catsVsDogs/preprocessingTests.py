import cv2
from matplotlib import pyplot as plt
import os
import numpy as np


TRAIN_DIR_RAW = 'data/train'
TRAIN_DIR_PROCESSED = 'data/processed/train'

TEST_DIR_RAW = 'data/test'
TEST_DIR_PROCESSED = 'data/processed/test'

RESIZE_WIDTH, RESIZE_HEIGHT = 300, 300

RESIZE_DIM = (RESIZE_WIDTH, RESIZE_HEIGHT)
# os.mkdir('data/processed')

trainNames = os.listdir(TRAIN_DIR_RAW)

fileName = trainNames[10]
filePath = TRAIN_DIR_RAW + '/' + fileName
img = cv2.imread(filePath, 0) #Read the image in black n white




imgResized = cv2.resize(img, dsize = RESIZE_DIM)
sobelY = cv2.Sobel(imgResized, cv2.CV_64F, 0, 1, ksize = 3)
sobelX = cv2.Sobel(imgResized, cv2.CV_64F, 1, 0, ksize = 3)

imgEdge = np.sqrt(sobelX ** 2 + sobelY ** 2).astype('uint8')
imgEdgeCanny = cv2.Canny(imgResized, 30, 150)







imgSmooth = cv2.GaussianBlur(imgResized, ksize = (3, 3), sigmaX = 2, sigmaY = 2)
sobelY = cv2.Sobel(imgSmooth, cv2.CV_64F, 0, 1, ksize = 3)
sobelX = cv2.Sobel(imgSmooth, cv2.CV_64F, 1, 0, ksize = 3)

imgSmoothEdge = np.sqrt(sobelX ** 2 + sobelY ** 2).astype('uint8')
imgSmoothEdgeCanny = cv2.Canny(imgSmooth, 60, 150)




fig, ax = plt.subplots(3, 2, figsize=(35,35))

ax[0, 0].imshow(imgResized, cmap='gray')
ax[0, 1].imshow(imgSmooth, cmap='gray')
ax[1, 0].imshow(imgEdge, cmap='gray')
ax[1, 1].imshow(imgSmoothEdge, cmap='gray')
ax[2, 0].imshow(imgEdgeCanny, cmap='gray')
ax[2, 1].imshow(imgSmoothEdgeCanny, cmap='gray')
plt.show()