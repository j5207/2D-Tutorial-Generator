from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import random
import math


AUGMENT = True
BATCH_SIZE = 1
output_vector = [8,6,1]
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = h_wid*out_pool_size[i] - previous_conv_size[0]
        w_pad = w_wid*out_pool_size[i] - previous_conv_size[1]
        pad = F.pad(previous_conv, (0, int(w_pad), int(h_pad), 0))
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid))
        x = maxpool(pad)
        if(i == 0):
            spp = x.view(num_sample,-1)
            #print("spp size:",spp.size())
        else:
            #print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
        #self.fc1 = nn.Linear(20*47*22, 100)
        self.fc1 = nn.Linear(4040, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
        x = self.conv3(x)
        spp = spatial_pyramid_pool(x,int(x.size(0)),[int(x.size(2)),int(x.size(3))],output_vector)
        print(spp.size())
        x = F.relu(self.fc1(spp))
        #x = x.view(-1, 20*47*22)
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class CovnetDataset(Dataset):
    def __init__(self, reader, transforms=None):
        self.reader = reader
        self.transform = transforms
    def __getitem__(self, item):
        image_tuple = self.reader[item]
        img1_dir = image_tuple[0]
        img1 = Image.open(img1_dir)
        label = float(image_tuple[1])
        result = torch.from_numpy(np.array([label], dtype=float))

        if AUGMENT:
            rotate_range = random.uniform(-180, 180)
            translation_range = random.uniform(-10, 10)
            scale_range = random.uniform(0.7, 1.3)
            if np.random.random() < 0.7:
                img1 = img1.rotate(rotate_range)
            if np.random.random() < 0.7:
                 img1 = img1.transform((img1.size[0], img1.size[1]), Image.AFFINE, (1, 0, translation_range, 0, 1, translation_range))
            if np.random.random() < 0.5:
                img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() < 0.5:
                img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            # if np.random.random() < 0.7:
            #     img1 = img1.resize((int(200 * scale_range), int(100 * scale_range)))
            #     half_the_width = img1.size[0] / 2
            #     half_the_height = img1.size[1] / 2
            #     img1 = img1.crop((half_the_width - 100,
            #             half_the_height - 100,
            #             half_the_width + 50,
            #             half_the_height + 50))
        img1 = self.transform(img1)
        return (img1, result)
    def __len__(self):
        return len(self.reader)


