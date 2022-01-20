import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import torchvision.transforms as transforms

import os
from torch.utils.tensorboard import SummaryWriter
import cv2
import matplotlib.pyplot as plt
import numpy as np



TRAIN_DIR = 'data/processed/train'
TEST_DIR = 'data/processed/test'
EPOCHS = 10
class Dataset(Dataset) :
    def __init__(self, PATH) : 
        self.PATH = PATH
        self.fileNames = os.listdir(self.PATH)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.PATH + '/' + self.fileNames[idx], 0)
        return img
    
    def __len__(self) : 
        return len(self.fileNames)


#Iniitiate log writer on tensorboard
logWriter = SummaryWriter("runs")


transformTrain = transforms.Compose(
    [
     transforms.RandomRotation(degrees = 30), 
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor()
    ])
transformTest = transforms.Compose(
    [transforms.ToTensor()
    ])



#Instantiate datasets for both train and test set
trainSet = Dataset(TRAIN_DIR)
testSet = Dataset(TEST_DIR)



batchSize = 16
trainLoader = DataLoader(dataset = trainSet, batch_size = batchSize, shuffle = True)
testLoader = DataLoader(dataset = testSet, batch_size = batchSize, shuffle = True)






itr = iter(trainLoader)

# fig, ax = plt.subplots(4, 4, figsize=(5,5))

# for i in range(4) :
#     for j in range(4) :
#         ax[i, j].imshow(itr.next()[i * 4 + j])

# plt.show()




class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding = 1)
        self.conv2 = nn.Conv2d(6, 16, 3, stride=1, padding = 1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding = 1)
        self.conv4 = nn.Conv2d(32, 16, 1, stride=1, padding = 0)
        self.batchNorm1 = nn.BatchNorm2d(16, affine=False)
        self.conv5 = nn.Conv2d(16, 8, 3, stride=1, padding = 1)
        self.conv6 = nn.Conv2d(6, 16, 3, stride=1, padding = 1)
        self.conv7 = nn.Conv2d(16, 32, 3, stride=1, padding = 1)
        self.conv8 = nn.Conv2d(32, 16, 1, stride=1, padding = 0)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1024, 512)  # 5*5 from image dimension
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim = 1)