import torch
from torch.utils.data import Dataset
import os
from torch.utils.tensorboard import SummaryWriter


class Dataset(Dataset) :
    def __init__(self, PATH) : 
        self.PATH = PATH
        self.fileNames = os.listdir(self.PATH)
    
    def __getitem__(self, idx):
        image_bgr = cv2.imread(self.PATH + '/' + self.fileNames[idx])
        image_rgb = image_bgr[ : , : , ::-1]
        return image_rgb
    
    def __len__(self) : 
        return len(self.fileNames)




logWriter = SummaryWriter("runs")