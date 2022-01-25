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
#LABELS : cat -> 0, dog -> 1
class Dataset(Dataset) :
    def __init__(self, PATH) : 
        self.PATH = PATH
        self.fileNames = os.listdir(self.PATH)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.PATH + '/' + self.fileNames[idx], 0)
        img = transforms.functional.to_tensor(img)
        img = img.reshape(1, 200, 200)
        label = 0 if self.fileNames[idx][ : 3] == 'cat' else 1
        return (img, label)
    
    def __len__(self) : 
        return len(self.fileNames)


#Iniitiate log writer on tensorboard
logWriter = SummaryWriter("runs")


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
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding = 1)
        self.conv4 = nn.Conv2d(64, 10, 1, stride=1, padding = 0)


        self.batchNorm1 = nn.BatchNorm2d(32, affine=False)

        self.conv5 = nn.Conv2d(10, 20, 3, stride=1, padding = 1)
        self.conv6 = nn.Conv2d(20, 40, 3, stride=1, padding = 1)
        self.conv7 = nn.Conv2d(40, 80, 3, stride=1, padding = 1)
        self.conv8 = nn.Conv2d(80, 16, 1, stride=1, padding = 0)

        self.conv9 = nn.Conv2d(16, 32, 3, stride=1, padding = 1)
        self.conv10 = nn.Conv2d(32, 64, 3, stride=1, padding = 1)
        self.conv11 = nn.Conv2d(64, 128, 3, stride=1, padding = 1)
        self.conv12 = nn.Conv2d(128, 32, 1, stride=1, padding = 0)

        self.conv13 = nn.Conv2d(32, 64, 3, stride=1, padding = 1)
        self.conv14 = nn.Conv2d(64, 64, 3, stride=1, padding = 1)
        self.conv16 = nn.Conv2d(64, 32, 1, stride=1, padding = 0)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4608, 512)  # 5*5 from image dimension
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x) :
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.max_pool2d(x, 2)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = torch.max_pool2d(x, 2)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = torch.max_pool2d(x, 2)


        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv16(x)
        x = torch.max_pool2d(x, 2)


        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


model = Net()
# optimizer = optim




def makeTrainStep(model,optimizer, lossFn ) :
    def trainStep(xTrain, yTrain) :
        model.train()
        optimizer.zero_grad()
        yPred = model(xTrain)

        loss = lossFn(yPred, yTrain)
        loss.backward()
        optimizer.step()

        return loss.detach()

    return trainStep





losses = []

epoch = 0

nEpochs = 2

learningRate = 0.001

lossFn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learningRate)

trainStep = makeTrainStep(model, optimizer, lossFn)



while epoch < nEpochs :
    
    
    print('Trainig pass : ', epoch + 1, ' of ', nEpochs, end = ' ')
    runningLoss = torch.zeros(1)
    for i, data in enumerate(trainLoader, 0) :
        xBatch, yBatch = data
        loss = trainStep(xBatch, yBatch)

        runningLoss += loss.item()


    logWriter.add_scalar('Loss (log)', torch.log(runningLoss), epoch)
    losses.append(torch.log(runningLoss))

    epoch += 1
    print('Complete')