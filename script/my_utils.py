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
import cv2

AUGMENT = True
BATCH_SIZE = 4


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20 * 47 * 22, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        # super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        # self.batchnorm1 = nn.BatchNorm2d(10)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.batchnorm2 = nn.BatchNorm2d(20)
        # self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
        # self.fc1 = nn.Linear(7560, 1000)
        # self.fc2 = nn.Linear(1000, 100)
        # self.fc3 = nn.Linear(100, 50)
        # self.fc4 = nn.Linear(50, 20)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
        x = x.view(-1, 20 * 47 * 22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # x = self.batchnorm1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        # x = self.batchnorm2(F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2)))
        # x = F.relu(F.max_pool2d(F.dropout2d(self.conv3(x)), 2))
        # #x = self.conv3(x)
        # #spp = spatial_pyramid_pool(x,int(x.size(0)),[int(x.size(2)),int(x.size(3))],output_vector)
        # #print(spp.size())
        # #x = F.relu(self.fc1(spp))
        # x = x.view(-1, 7560)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
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
                img1 = img1.transform((img1.size[0], img1.size[1]), Image.AFFINE,
                                      (1, 0, translation_range, 0, 1, translation_range))
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
#used to determine whether the two pipes are about the merge, the input is two lists contain coordinates of two center point [x1,y1] [x2,y2]
def if_connecting(target_center_left, target_center_right):
    distant = lambda x1, y1, x2, y2: sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    threhold = 10; #need to decide
    if distant(target_center_left[0],target_center_left[1],target_center_right[0],target_center_right[1])<threhold:
        return True
    else:
        return False

#return the center of selected points with different weights. Input is the selected points and their weight. Example: [[1,2],[3,4]] [1,2]
def find_center(counters,weights):
    counter_num = len(counters)
    cx,cy,w = 0,0,0,0
    for i in range(1,counter_num+1):
        cx = cx + counters[i-1][0] * weights[i-1]
        cy = cy + counters[i-1][1] * weights[i-1]
    cx = cx/sum(weights)
    cy = cy/sum(weights)
    return [cx,cy]

#used when two pipes is merging, return the direction of combination. center_target is the coordinate of combining sides, like [1,2] centers_supporting is a list of other center points coordniate in frame. Weights is a list containg weight of the sides.
def side_dirction_finder(center_target_left, center_target_right, centers_supporting_left, centers_supporting_right, weights_left, weights_right):
    center_all_left = find_center(centers_supporting_left, weights_left)
    center_all_right = find_center(centers_supporting_right, weights_right)
    # the equation of the target line is presented as Ax + By + C =0
    A = center_target_right[1] - center_target_left[1]
    B = center_target_left[0] - center_target_right[0]
    C = center_target_right[0] * center_target_left[1] - center_target_left[0] * center_target_right[1]
    threhold=0  #need to decide
    if (A * center_all_left[0] + B * center_all_left[1] + C) * (A * center_all_right[0] + B * center_all_right[1] + C) > threhold:
        return "Postive"
    else:
        return "Negitive"

def side_finder(frame, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = [np.array([172, 62, 0]), np.array([180, 255, 255])]
    blue = [np.array([100, 118, 112]), np.array([113, 255, 255])]
    if color == 'red':
        mask = cv2.inRange(hsv, *red)
    elif color == 'blue':
        mask = cv2.inRange(hsv, *blue)
    else:
        raise NameError('red or blue')
    kernal = np.ones((3, 3), "uint8")
    mask = cv2.dilate(mask, kernal)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center_list = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100 and hierarchy[0, i, 3] == -1:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # cv2.circle(frame, (cx, cy), 10, [0, 255, 0])
            center_list.append((cx, cy))
    return center_list


def test_insdie(point, boundingbox_list):
    cx, cy = point
    for i, boundingbox in enumerate(boundingbox_list):
        x, y, w, h = boundingbox
        if cx > x and cx < x + w and cy > y and cy < y + h:
            return i


class cache():
    def __init__(self, length):
        self.list = []
        self.length = length
        self.full = False

    def append(self, data):
        if len(self.list) < self.length:
            self.list.append(data)
        else:
            del self.list[0]
            self.append(data)
            self.full = True

    def clear(self):
        self.list = []
        self.full = False

# from __future__ import division
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import os
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import torch
# from PIL import Image
# from torchvision import transforms
# from torch.autograd import Variable
# import numpy as np
# import random
# import math


# AUGMENT = True
# BATCH_SIZE = 1
# output_vector = [50, 20, 10, 8, 6, 1]
# def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
#     '''
#     previous_conv: a tensor vector of previous convolution layer
#     num_sample: an int number of image in the batch
#     previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
#     out_pool_size: a int vector of expected output size of max pooling layer

#     returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
#     '''
#     # print(previous_conv.size())
#     for i in range(len(out_pool_size)):
#         h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
#         w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
#         h_pad = h_wid*out_pool_size[i] - previous_conv_size[0]
#         w_pad = w_wid*out_pool_size[i] - previous_conv_size[1]
#         pad = F.pad(previous_conv, (0, int(w_pad), int(h_pad), 0))
#         maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid))
#         x = maxpool(pad)
#         if(i == 0):
#             spp = x.view(num_sample,-1)
#             #print("spp size:",spp.size())
#         else:
#             #print("size:",spp.size())
#             spp = torch.cat((spp,x.view(num_sample,-1)), 1)
#     return spp

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
#         #self.fc1 = nn.Linear(20*47*22, 100)
#         self.fc1 = nn.Linear(40 * sum(map(lambda x: x*x, output_vector)), 100)
#         self.fc2 = nn.Linear(100, 50)
#         self.fc3 = nn.Linear(50, 20)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
#         x = self.conv3(x)
#         spp = spatial_pyramid_pool(x,int(x.size(0)),[int(x.size(2)),int(x.size(3))],output_vector)
#         #print(spp.size())
#         x = F.relu(self.fc1(spp))
#         #x = x.view(-1, 20*47*22)
#         #x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.softmax(x, dim=1)

# class CovnetDataset(Dataset):
#     def __init__(self, reader, transforms=None):
#         self.reader = reader
#         self.transform = transforms
#     def __getitem__(self, item):
#         image_tuple = self.reader[item]
#         img1_dir = image_tuple[0]
#         img1 = Image.open(img1_dir)
#         label = float(image_tuple[1])
#         result = torch.from_numpy(np.array([label], dtype=float))

#         if AUGMENT:
#             rotate_range = random.uniform(-180, 180)
#             translation_range = random.uniform(-10, 10)
#             scale_range = random.uniform(0.7, 1.3)
#             if np.random.random() < 0.7:
#                 img1 = img1.rotate(rotate_range)
#             if np.random.random() < 0.7:
#                  img1 = img1.transform((img1.size[0], img1.size[1]), Image.AFFINE, (1, 0, translation_range, 0, 1, translation_range))
#             if np.random.random() < 0.5:
#                 img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
#             if np.random.random() < 0.5:
#                 img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
#             # if np.random.random() < 0.7:
#             #     img1 = img1.resize((int(200 * scale_range), int(100 * scale_range)))
#             #     half_the_width = img1.size[0] / 2
#             #     half_the_height = img1.size[1] / 2
#             #     img1 = img1.crop((half_the_width - 100,
#             #             half_the_height - 100,
#             #             half_the_width + 50,
#             #             half_the_height + 50))
#         img1 = self.transform(img1)
#         return (img1, result)
#     def __len__(self):
#         return len(self.reader)


