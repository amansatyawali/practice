import cv2
from matplotlib import pyplot as plt
import os
import numpy as np


TRAIN_DIR_RAW = 'data/train'
TRAIN_DIR_PROCESSED = 'data/processed/train'

TEST_DIR_RAW = 'data/test'
TEST_DIR_PROCESSED = 'data/processed/test'

RESIZE_WIDTH, RESIZE_HEIGHT = 200, 200

RESIZE_DIM = (RESIZE_WIDTH, RESIZE_HEIGHT)



def processImage(img) :
  imgSmooth = cv2.GaussianBlur(img, ksize = (5, 5), sigmaX = 2, sigmaY = 2) #Applying blur to the image for noise reduction
  #Get Edges by the Sobel kernel
  # sobelY = cv2.Sobel(imgSmooth, cv2.CV_64F, 0, 1, ksize = 3)
  # sobelX = cv2.Sobel(imgSmooth, cv2.CV_64F, 1, 0, ksize = 3)
  # imgSmoothEdge = np.sqrt(sobelX ** 2 + sobelY ** 2).astype('uint8')
  imgSmoothCanny = cv2.Canny(imgSmooth, 30, 100)              #Applying Canny edge detection
  imgResizedSmooth = cv2.resize(imgSmoothCanny, dsize = RESIZE_DIM)
  return imgResizedSmooth





os.mkdir('data/processed')
os.mkdir(TRAIN_DIR_PROCESSED)
os.mkdir(TEST_DIR_PROCESSED)

trainNames = os.listdir(TRAIN_DIR_RAW)
testNames = os.listdir(TEST_DIR_RAW)

for fileName in trainNames :
  filePath = TRAIN_DIR_RAW + '/' + fileName
  img = cv2.imread(filePath, 0) #Read the image in black n white
  processedImage = processImage(img)
  filePathOutput = TRAIN_DIR_PROCESSED + '/' + fileName
  cv2.imwrite(filePathOutput, processedImage)

print('Training set processed')

for fileName in testNames :
  filePath = TEST_DIR_RAW + '/' + fileName
  img = cv2.imread(filePath, 0) #Read the image in black n white
  processedImage = processImage(img)
  filePathOutput = TEST_DIR_PROCESSED + '/' + fileName
  cv2.imwrite(filePathOutput, processedImage)

print('Test set processed')