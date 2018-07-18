from __future__ import print_function
import numpy as np
import cv2
import time
import time
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
import random
import tqdm
from math import sqrt
import matplotlib.pyplot as plt
import Tkinter as tk
import threading
from utils import CovnetDataset, Net

LR = 0.0002
BATCH_SIZE = 4
GPU = True
COLLECT_TIME = 1.0
AUGMENT = False
EPOTH = 100
ONLY_TEST = False

distant = lambda (x1, y1), (x2, y2) : sqrt((x1 - x2)**2 + (y1 - y2)**2)

class temp_list():
    def __init__(self, length):
        self.list = []
        self.length = length
    
    def append(self, data):
        if len(self.list) < self.length:
            self.list.append(data)
        else:
            del self.list[0]
            self.append(data)


class object_detector(): 
    def __init__(self, start): 		
        self.start_time = start
        self.stored_flag = False
        self.trained_flag = False
        self.boxls = None
        self.count = 1
        self.path = "/home/intuitivecompting/Desktop/color/Smart-Projector/script/datasets/"
        self.file = open(self.path + "read.txt", "w")
        self.user_input = 0
        self.predict = None
        self.memory = temp_list(100)
        self.num_object = 2
        #self.thread1 = threading.Thread(target=self.thread_1, args=())
        #self.cap = cap

    # def thread_1(self):
    #     while True:
    #         self.update(self.cap, save=True)
    #         if cv2.waitKey(5) & 0xFF == 27:
    #             break


    def update(self, cap, save=False, train=False):
        self.boxls = []
        _, origin = cap.read()
        rect = self.camrectify(origin)

        #-------------warp the image---------------------#
        warp = self.warp(rect)
        #-------------segment the object----------------#
        hsv = cv2.cvtColor(warp,cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([57,145,0]), np.array([85,255,255]))
        hand_mask = cv2.inRange(hsv, np.array([118,32,0]), np.array([153,255,255]))
        hand_mask = cv2.dilate(hand_mask, kernel = np.ones((7,7),np.uint8))

        skin_mask = cv2.inRange(hsv, np.array([0,52,0]), np.array([56,255,255]))
        skin_mask = cv2.dilate(skin_mask, kernel = np.ones((5,5),np.uint8))

        thresh = 255 - green_mask
        thresh = cv2.subtract(thresh, hand_mask)
        thresh = cv2.subtract(thresh, skin_mask)
        #thresh = cv2.erode(thresh, kernel = np.ones((5,5),np.uint8))
        #cv2.imshow('thresh', thresh)
        draw_img1 = warp.copy()
        draw_img2 = warp.copy()
        self.train_img = warp.copy()
        #-------------get the bounding box--------------
        self.get_bound(draw_img1, thresh, hand_mask, only=False, visualization=True)
        #--------------get bags of words and training-------#
        if not ONLY_TEST:
            if not self.stored_flag:
                cv2.imshow('store', draw_img1)
                self.stored_flag = self.store(COLLECT_TIME)
            if self.stored_flag and not self.trained_flag: 
                cv2.destroyWindow('store')
                self.trained_flag = self.train()
            if self.trained_flag: 
                self.train(draw_img2, is_train=False)
                cv2.imshow('track', draw_img2)
        else:
            self.train(draw_img2, is_train=False)
            cv2.imshow('track', draw_img2)
        
    def get_bound(self, img, object_mask, hand_mask, only=True, visualization=True):
        (_,object_contours, object_hierarchy)=cv2.findContours(object_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        (_,hand_contours, hand_hierarchy)=cv2.findContours(hand_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        hand_m_ls = []
        object_m_ls = []
        if len(hand_contours) > 0:
            for i , contour in enumerate(hand_contours):
                area = cv2.contourArea(contour)
                if area>600 and area < 100000 and hand_hierarchy[0, i, 3] == -1:					
                    M = cv2.moments(contour)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    hand_m_ls.append((cx, cy))
        if len(object_contours) > 0:
            for i , contour in enumerate(object_contours):
                area = cv2.contourArea(contour)
                if area>600 and area < 100000 and object_hierarchy[0, i, 3] == -1:					
                    M = cv2.moments(contour)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    object_m_ls.append((cx, cy))
                    x,y,w,h = cv2.boundingRect(contour)
                    self.boxls.append([x, y, w, h])
        temp_i = []
        temp_j = []
        for (x3, y3) in hand_m_ls:
            for i in range(len(object_m_ls)):
                for j in range(i + 1, len(object_m_ls)):
                    x1, y1 = object_m_ls[i]
                    x2, y2 = object_m_ls[j]
                    d12 = distant((x1, y1), (x2, y2))
                    d13 = distant((x1, y1), (x3, y3))
                    d23 = distant((x2, y2), (x3, y3))
                    dis = d13 * d23 / d12
                    if dis < 60 and d12 < 140 and d13 < 100 and d23 < 100:
                        temp_i.append(i)
                        temp_j.append(j)
                        # print(dis, d12, d13, d23)

        if len(temp_i) > 0 and len(temp_j) > 0 and len(self.boxls) >= 1:
            for (i, j) in zip(temp_i, temp_j):
                if self.boxls[i] != 0 and self.boxls[j] != 0:
                    x, y = np.min([self.boxls[i][0], self.boxls[j][0]]), np.min([self.boxls[i][1], self.boxls[j][1]])
                    x_max, y_max = np.max([self.boxls[i][0] + self.boxls[i][2], self.boxls[j][0] + self.boxls[j][2]]), np.max([self.boxls[i][1] + self.boxls[i][3], self.boxls[j][1] + self.boxls[j][3]])         
                    w, h = x_max - x, y_max - y
                    self.boxls[i] = 0
                    self.boxls[j] = [x, y, w, h]
            
            self.boxls = filter(lambda a: a != 0, self.boxls)   

            #---------------sorting the list according to the x coordinate of each item
        if len(self.boxls) > 0:
            boxls_arr = np.array(self.boxls)
            self.boxls = boxls_arr[boxls_arr[:, 0].argsort()].tolist()
        for i in range(len(self.boxls)): 
            if visualization: 
                ind = max(range(len(self.boxls)), key=lambda i:self.boxls[i][2]*self.boxls[i][3])
                x,y,w,h = self.boxls[ind]
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(img,str(self.user_input),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))

   
        

    def warp(self, img):
        pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
        pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(640,480))
        return dst
    
    def store(self, store_time):
        if len(self.boxls) > 0:
            frame = self.train_img
            ind = max(range(len(self.boxls)), key=lambda i:self.boxls[i][2]*self.boxls[i][3])
        #-------------capturing img for each of item--------------#
            x,y,w,h = self.boxls[ind]
            temp = frame[y:y+h, x:x+w, :]
            img_dir = os.path.join(self.path + "image", str(self.count) + ".jpg")
            self.createFolder(self.path + "image")
            cv2.imwrite(img_dir, temp)
            self.count += 1 
            self.file.write(img_dir + " " + str(self.user_input) + "\n")
            #------------------storing-------------------- 
            if time.time() - self.start_time < store_time:
                print('output imgs ' + str(self.count) + 'img' )
                return False
            #-----------------get to the next item-----------
            else:
                print("previous label: {} \n".format(self.user_input))
                self.user_input = int(raw_input("please enter label, or enter -1 as finish \n"))
                if self.user_input != -1:
                    self.start_time = time.time()
                    self.num_object += 1
                    return False
                else: 
                    self.file.close()
                    print('finish output')
                    return True
        else:
            return False


    
    def train(self, draw_img=None, is_train=True):
        self.predict = {}
        if not GPU:
            net = Net()
        else:
            net = Net().cuda()
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=LR)
        reader_train = self.reader(self.path, "read.txt")
        trainset = CovnetDataset(reader=reader_train, transforms=transforms.Compose([transforms.Resize((200, 200)),
                                                                                            transforms.ToTensor()
                                                                                       ]))
        trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#-----------------------------------training----------------------------------------------------------------        
        if is_train:
            loss_ls = []
            count = 0
            count_ls = []
            t = tqdm.trange(EPOTH, desc='Training')
            temp = 0
            for epoch in t:  # loop over the dataset multiple times
                running_loss = 0.0
                i = 0
                for data in trainloader:
                    # get the inputs
                    inputs, labels = data
                    if GPU:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    inputs, labels = Variable(inputs), Variable(labels.long())
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = F.cross_entropy(outputs, labels.view(1, -1)[0])
                    loss.backward()
                    optimizer.step()
                    t.set_description('loss=%g' %(temp))

                    loss_ls.append(loss.item())
                    count += 1
                    count_ls.append(count)
                    
                    running_loss += loss.item()                    
                    if i % 20 == 19:   
                        temp = running_loss/20
                        running_loss = 0.0
                    i += 1
            plt.plot(count_ls, loss_ls)
            plt.show(block=False)
            print('Finished Training')
            torch.save(net.state_dict(), f=self.path + 'model')
            return True
#---------------------------------testing-----------------------------------------------
        else:
            self.predict = []
            net.load_state_dict(torch.load(f=self.path + 'model'))
            num_object = len(self.boxls)
            frame = self.train_img
            preprocess = transforms.Compose([transforms.Resize((200, 200)),
                                                        transforms.ToTensor()])
            for i in range(num_object):
                x,y,w,h = self.boxls[i]
                temp = frame[y:y+h, x:x+w, :]
                temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
                image = Image.fromarray(temp)
                img_tensor = preprocess(image)
                img_tensor.unsqueeze_(0)
                img_variable = Variable(img_tensor).cuda()
                #print("i:{} \n vector:{}".format(i, np.max(net(img_variable).cpu().data.numpy()[0])))
                if np.max(net(img_variable).cpu().data.numpy()[0]) > 0.97:
                    out = np.argmax(net(img_variable).cpu().data.numpy()[0])
                else:
                    out = -1
                cv2.rectangle(draw_img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(draw_img,str(out),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
                self.predict.append(((x, y, w, h), out))
            self.memory.append(self.predict)
            #print(len(self.memory.list))
    #---------------------------------merge---------------------------------------#
    
    def merge(self):
        if len(self.memory.list) < 100:
            #print("pass")
            pass
        else: 
            #print("nm", self.num_object)
            num = len(filter(lambda a: len(a) < self.num_object, self.memory.list))
            #print(num)
            # if num > 70:
                #print("######################merge##########################")
            
            
                
                
                




    @staticmethod
    def reader(path, mode):
        path= ''.join([path, mode])
        f = open(path)
        data = f.readlines()
        data_ls = []
        for line in data:
            objects = line.split()
            data_ls.append(objects)
        return data_ls

    @staticmethod    
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

    @staticmethod
    def camrectify(frame):
        mtx = np.array([
            [509.428319, 0, 316.944024],
            [0.000000, 508.141786, 251.243128],
            [0.000000, 0.000000, 1.000000]
        ])
        dist = np.array([
            0.052897, -0.155430, 0.005959, 0.002077, 0.000000
        ])
        return cv2.undistort(frame, mtx, dist)



def main():
    cap=cv2.VideoCapture(0)
    start_time = time.time()
    detector = object_detector(start_time)
    while 1:
        detector.update(cap, save=True)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    # detector.thread1.start()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




 # def get_bound(self, img, mask, only=True, visualization=True):
    #     (_,contours, hierarchy)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) > 0:
    #         for i , contour in enumerate(contours):
    #             area = cv2.contourArea(contour)
    #             if area>600 and area < 100000 and hierarchy[0, i, 3] == -1:					
    #                 M = cv2.moments(contour)
    #                 cx = int(M['m10']/M['m00'])
    #                 cy = int(M['m01']/M['m00'])
    #                 rect = cv2.minAreaRect(contour)
    #                 box = np.int0(cv2.boxPoints(rect))
    #                 x,y,w,h = cv2.boundingRect(contour)
    #                 self.boxls.append([x, y, w, h])
    #         #---------------sorting the list according to the x coordinate of each item
    #         if len(self.boxls) > 0:
    #             boxls_arr = np.array(self.boxls)
    #             self.boxls = boxls_arr[boxls_arr[:, 0].argsort()].tolist()
    #         for i in range(len(self.boxls)): 
    #             if visualization: 
    #                 x,y,w,h = self.boxls[i]
    #                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    #                 cv2.putText(img,str(self.user_input),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))

    
        
    # def store(self, store_time):
    #     num_object = len(self.boxls)
    #     frame = self.train_img
    #     #-------------capturing img for each of item--------------#
    #     for i in range(num_object):
    #         x,y,w,h = self.boxls[i]
    #         temp = frame[y:y+h, x:x+w, :]
    #         img_dir = os.path.join(self.path + "image", str(self.count) + ".jpg")
    #         cv2.imwrite(img_dir, temp)
    #         self.count += 1 
    #         self.file.write(img_dir + " " + str(i) + "\n") 
    #     if time.time() - self.start_time < store_time:
    #         print('output imgs ' + str(self.count) + 'img' )
    #         return False
    #     else:
    #         self.file.close()
    #         print('finish output')
    #         return True

    # class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
#         self.fc1 = nn.Linear(40*21*21, 100)
#         self.fc2 = nn.Linear(100, 50)
#         self.fc3 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
#         x = F.relu(F.max_pool2d(F.dropout2d(self.conv3(x)), 2))
#         #print(x.size())
#         x = x.view(-1, 40*21*21)
#         x = F.relu(self.fc1(x))
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
