import os
import numpy as np
import shutil

TEST_RATIO = 0.2

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'


dataSetFileNames = os.listdir(TRAIN_DIR) # get the names of all the files in the train folder so that it is easier to move data by name
dataSetLen = len(dataSetFileNames)

os.mkdir(TEST_DIR)

testLen = int(TEST_RATIO * dataSetLen)

idx = np.arange(dataSetLen) #Creating a numpy array from 0 to dataSetLen as index for each datapoint

np.random.shuffle(idx) #Shuffle the elements of idx

testIdx = idx[ : testLen] #Getting indexes for the test set (since the array was shuffled, output will be a random selection from all the indexes)

for i in testIdx :
    currFilePath = TRAIN_DIR + '/' + dataSetFileNames[i]
    newFilePath = TEST_DIR + '/' + dataSetFileNames[i]
    shutil.copyfile(currFilePath, newFilePath)        #Copy each file with the test index to the test folder
    os.remove(currFilePath)                           #Delete the file from the train directory
