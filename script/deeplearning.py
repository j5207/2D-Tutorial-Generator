from __future__ import print_function
import numpy as np
import cv2
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image, ImageTk
from torchvision import transforms
from torch.autograd import Variable
import random
import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
from utils import CovnetDataset, Net, BATCH_SIZE, side_finder, test_insdie
from hand_tracking import hand_tracking
from shapely.geometry import Polygon


LR = 0.0003

#GPU = False
GPU = torch.cuda.is_available()
EPOTH = 50

# MODE could be 'train', 'test',  'all'
MODE = 'train'

distant = lambda (x1, y1), (x2, y2) : sqrt((x1 - x2)**2 + (y1 - y2)**2)

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

class node():
    num_instance = 0
    instance_list = []
    pair_list = []
    def __init__(self, (id1, id2), (side1, side2),outcome):
        self.pair = (id1, id2)
        self.sides = (side1, side2)
        self.outcome = outcome
        node.num_instance += 1
        node.instance_list.append(self)
        node.pair_list.append((self.pair, self.sides,self.outcome))
        print(self)

    def __str__(self):
        return "this is the {} node, contains {}, sides {}".format(node.num_instance, self.pair, self.sides)
        


class object_detector(): 
    def __init__(self, start): 	
        self.cap = cv2.VideoCapture(0)	
        self.start_time = start

        self.stored_flag = False
        self.trained_flag = False
        self.milstone_flag = False
        self.incremental_train_flag = False
        self.tracking_flag = False

        self.boxls = None
        self.count = 1
        self.path = "/home/intuitivecompting/Desktop/color/Smart-Projector/script/datasets/"
        if MODE == 'all':
            self.file = open(self.path + "read.txt", "w")
            self.milestone_file = open(self.path + "mileston_read.txt", "w")
        self.user_input = 0
        self.predict = None
        self.memory = cache(10)
        self.memory1 = cache(10)

        self.node_sequence = []
        #-----------------------create GUI-----------------------#
        self.gui_img = np.zeros((130,640,3), np.uint8)
        cv2.circle(self.gui_img,(160,50),30,(255,0,0),-1)
        cv2.putText(self.gui_img,"start",(130,110),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,0,0))
        cv2.circle(self.gui_img,(320,50),30,(0,255,0),-1)
        cv2.putText(self.gui_img,"stop",(290,110),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,0))
        cv2.circle(self.gui_img,(480,50),30,(0,0,255),-1)
        cv2.putText(self.gui_img,"quit",(450,110),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
        cv2.namedWindow('gui_img')
        cv2.namedWindow('gui_img1')
        cv2.setMouseCallback('gui_img',self.gui_callback)
        cv2.setMouseCallback('gui_img1',self.gui_callback)
        #-----------------------Training sign--------------#
        self.training_surface = np.ones((610,640,3), np.uint8) * 255
        cv2.putText(self.training_surface,'Training...',(120,300),cv2.FONT_HERSHEY_SIMPLEX, 3.0,(255,192,203), 5)
        #----------------------new coming item id------------------#
        self.new_come_id = None
        self.old_come_id = None
        self.new_come_side = None
        self.old_come_side = None
        self.new_coming_lock = True
        self.once_lock = True
        #---------------------set some flag-------------------#
        self.storing = None
        self.quit = None
        self.once = True
        #---------------------set gui image----------------------#
        self.temp_surface = None
        #----------------------for easlier developing-----------------#
        if MODE == 'test':
            if not GPU:
                self.net = Net()
            else:
                self.net = Net().cuda()
            self.net.load_state_dict(torch.load(f=self.path + 'model'))
            self.user_input = 10


    def update(self, save=True, train=False):
        
        self.boxls = []
        OK, origin = self.cap.read()
        if OK:
            rect = self.camrectify(origin)

            #-------------warp the image---------------------#
            warp = self.warp(rect)

            #-------------segment the object----------------#
            hsv = cv2.cvtColor(warp,cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array([63,101,61]), np.array([86,255,255]))
            # green_mask = cv2.inRange(hsv, np.array([45,90,29]), np.array([85,255,255]))
            hand_mask = cv2.inRange(hsv, np.array([118,32,0]), np.array([153,255,255]))
            hand_mask = cv2.dilate(hand_mask, kernel = np.ones((7,7),np.uint8))

            skin_mask = cv2.inRange(hsv, np.array([0,36,0]), np.array([17,255,255]))
            skin_mask = cv2.dilate(skin_mask, kernel = np.ones((7,7),np.uint8))

            
            
            thresh = 255 - green_mask
            thresh = cv2.subtract(thresh, hand_mask)
            thresh = cv2.subtract(thresh, skin_mask)
            thresh[477:, 50:610] = 0
            cv2.imshow('afg', thresh)
            draw_img1 = warp.copy()
            draw_img2 = warp.copy()
            draw_img3 = warp.copy()
            self.train_img = warp.copy()
            #-------------get the bounding box--------------
            self.get_bound(draw_img1, thresh, hand_mask, only=False, visualization=True)
            #--------------get bags of words and training-------#
            if MODE == 'all':
                #----------------------------storing image for each item---------#
                if not self.stored_flag:
                    self.temp_surface = np.vstack((draw_img1, self.gui_img))                    
                    self.stored_flag = self.store()
                    cv2.imshow('gui_img', self.temp_surface)
                #--------------------------training, just once------------------#
                if self.stored_flag and not self.trained_flag:  
                    cv2.destroyWindow('gui_img')
                    #cv2.imshow('training', self.training_surface)
                    self.trained_flag = self.train()
                #------------------------assembling and saving milstone---------#
                if self.trained_flag and not self.milstone_flag: 
                    self.test(draw_img2)
                    self.temp_surface = np.vstack((draw_img2, self.gui_img))
                    cv2.imshow('gui_img1', self.temp_surface)
                #-----------------------training saved milstone image---------#
                if self.milstone_flag and not self.incremental_train_flag:
                    cv2.destroyWindow('gui_img1')
                    self.incremental_train_flag = self.train(is_incremental=True)
                #-----------------------finalized tracking------------------#
                if self.incremental_train_flag and not self.tracking_flag:
                    self.test(draw_img3, is_tracking=True)
                    cv2.imshow('tracking', draw_img3)
            elif MODE == 'test':
                self.test(draw_img2)
                self.temp_surface = np.vstack((draw_img2, self.gui_img))
                cv2.imshow('gui_img', self.temp_surface)
                #cv2.imshow('track', draw_img2)
                #-----------------------training saved milstone image---------#
                if self.milstone_flag and not self.incremental_train_flag:
                    cv2.destroyWindow('gui_img')
                    self.incremental_train_flag = self.train(is_incremental=True)
                #-----------------------finalized tracking------------------#
                if self.incremental_train_flag and not self.tracking_flag:
                    self.test(draw_img3, is_tracking=True)
                    cv2.imshow('gui_img1', draw_img3)
            elif MODE == 'train':
                if not self.trained_flag:  
                    #cv2.destroyWindow('gui_img')
                    #cv2.imshow('training', self.training_surface)
                    self.trained_flag = self.train()
                #------------------------assembling and saving milstone---------#
                if self.trained_flag and not self.milstone_flag: 
                    self.test(draw_img2)
                    self.temp_surface = np.vstack((draw_img2, self.gui_img))
                    cv2.imshow('gui_img1', self.temp_surface)
                #-----------------------training saved milstone image---------#
                if self.milstone_flag and not self.incremental_train_flag:
                    cv2.destroyWindow('gui_img1')
                    self.incremental_train_flag = self.train(is_incremental=True)
                #-----------------------finalized tracking------------------#
                if self.incremental_train_flag and not self.tracking_flag:
                    self.test(draw_img3, is_tracking=True)
                    cv2.imshow('tracking', draw_img3)
        
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
                if area>100 and area < 100000 and object_hierarchy[0, i, 3] == -1:					
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
                    # dis = d13 * d23 / d12
                    # if dis < 60 and d12 < 140 and d13 < 100 and d23 < 100:
                    #     temp_i.append(i)
                    #     temp_j.append(j)
                    dis = self.get_k_dis((x1, y1), (x2, y2), (x3, y3))
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

     
    def gui_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK and (self.temp_surface[y, x] == np.array([255, 0, 0])).all() and not self.storing:
            self.count = 1
            self.user_input += 1
            self.storing = True
            if self.user_input > 10:
                if self.once:
                    temp_node = node((self.new_come_id, self.old_come_id), (self.new_come_side, self.old_come_side),self.user_input)
                    self.once = False
                else:
                    temp_node = node((self.new_come_id, self.user_input - 1), (self.new_come_side, self.old_come_side), self.user_input)
                self.node_sequence.append(temp_node)
            print("start")
        if event == cv2.EVENT_LBUTTONDBLCLK and (self.temp_surface[y, x] == np.array([0, 255, 0])).all() and self.storing:
            self.storing = False
            self.new_coming_lock = True
            print("stop")
        if event == cv2.EVENT_LBUTTONDBLCLK and (self.temp_surface[y, x] == np.array([0, 0, 255])).all():
            self.storing = None
            self.quit = True
            print("quit")
        # if event == cv2.EVENT_LBUTTONDBLCLK and (self.temp_surface[y, x] == np.array([255, 0, 255])).all():
        #     self.saving_milstone = True
        #     self.user_input += 1

    def store(self, is_milestone=False):
        # if is_milestone:
        #     file = self.milestone_file
        #     img_dir = os.path.join(self.path + "milestone_image", str(self.count) + ".jpg")
        #     self.createFolder(self.path + "milestone_image")
        # else:
        if is_milestone:
            self.file = open(self.path + "read.txt", "a")
            img_dir = os.path.join(self.path + "image", "milstone" + str(self.user_input)+str(self.count) + ".jpg")
        else:
            img_dir = os.path.join(self.path + "image", str(self.user_input)+str(self.count) + ".jpg")
        file = self.file
        self.createFolder(self.path + "image")
        if self.quit:
                file.close()
                print('finish output')               
                return True
        if len(self.boxls) > 0:
            if self.storing:
                cv2.putText(self.temp_surface,"recording",(450,50),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255), 2)
                frame = self.train_img
                ind = max(range(len(self.boxls)), key=lambda i:self.boxls[i][2]*self.boxls[i][3])
            #-------------capturing img for each of item--------------#
                x,y,w,h = self.boxls[ind]
                temp = frame[y:y+h, x:x+w, :]
                
                cv2.imwrite(img_dir, temp)         
                file.write(img_dir + " " + str(self.user_input) + "\n")
                if self.count % 100 == 0:
                    print('output imgs ' + str(self.count) + 'img' )
                self.count += 1 
                return False
            #-----------------get to the next item-----------    
        else:
            return False
        

    
        
    def train(self, is_incremental=False):
        start_time = time.time()
        if not is_incremental:
            reader_train = self.reader(self.path, "read.txt")
            if not GPU:
                self.net = Net()
            else:
                self.net = Net().cuda()
        else:
            if not GPU:
                self.net = Net()
            else:
                self.net = Net().cuda()
            reader_train = self.reader(self.path, "read.txt")
            self.net.load_state_dict(torch.load(f=self.path + 'model'))
        #optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9, nesterov=True)
        optimizer = optim.Adam(self.net.parameters(), lr=LR, weight_decay=0.1)
        schedule = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        trainset = CovnetDataset(reader=reader_train, transforms=transforms.Compose([transforms.Resize((200, 100)),
                                                                                            transforms.ToTensor()
                                                                                    ]))
        # trainset = CovnetDataset(reader=reader_train, transforms=transforms.Compose([transforms.Pad(30),
        #                                                                                      transforms.ToTensor()
        #                                                                              ]))
        trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#-----------------------------------training----------------------------------------------------------------        
        if True:
            loss_ls = []
            count = 0
            count_ls = []
            t = tqdm.trange(EPOTH, desc='Training')
            temp = 0
            for _ in t:  # loop over the dataset multiple times
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
                    outputs = self.net(inputs)
                    # print(outputs)
                    # print(labels.view(1, -1)[0])
                    loss = F.cross_entropy(outputs, labels.view(1, -1)[0])
                    loss.backward()
                    optimizer.step()
                    t.set_description('loss=%g' %(temp))

                    loss_ls.append(loss.item())
                    count += 1
                    count_ls.append(count)
                    
                    running_loss += loss.item()                    
                    if i % 10 == 9:   
                        temp = running_loss/10
                        running_loss = 0.0
                    i += 1
            plt.plot(count_ls, loss_ls)
            plt.show(block=False)
            print('Finished Training, using {} second'.format(int(time.time() - start_time)))
            
            self.quit = None
            
            if not is_incremental:
                self.user_input = 10
                torch.save(self.net.state_dict(), f=self.path + 'model')
            else:
                torch.save(self.net.state_dict(), f=self.path + 'milestone_model')
                pickle.dump(node.pair_list ,open("node.p", "wb"))
                try:
                    node_file = open(self.path + "node.txt", "w")
                    for pair in node.pair_list: 
                        node_file.write(str(pair[0]) + "" + str(pair[1]) + "\n")
                except:
                    print("fail to save")
            return True
#---------------------------------testing-----------------------------------------------
        
    def test(self, draw_img, is_tracking=False):
        self.predict = []
        net = self.net
        num_object = len(self.boxls)
        frame = self.train_img
        preprocess = transforms.Compose([transforms.Resize((200, 100)),
                                                    transforms.ToTensor()])
        # preprocess = transforms.Compose([transforms.Pad(30),
        #                                              transforms.ToTensor()])
        for i in range(num_object):
            x,y,w,h = self.boxls[i]
            temp = frame[y:y+h, x:x+w, :]
            temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
            image = Image.fromarray(temp)
            img_tensor = preprocess(image)
            img_tensor.unsqueeze_(0)
            img_variable = Variable(img_tensor).cuda()
            if GPU:
                img_variable = Variable(img_tensor).cuda()
                out = np.argmax(net(img_variable).cpu().data.numpy()[0])
            else:
                img_variable = Variable(img_tensor)
                out = np.argmax(net(img_variable).data.numpy()[0])
            # if np.max(net(img_variable).cpu().data.numpy()[0]) > 0.9:
            #     out = np.argmax(net(img_variable).cpu().data.numpy()[0])
            # else:
            #     out = -1
            cv2.rectangle(draw_img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(draw_img,str(out),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
            self.predict.append(((x, y, w, h), out))
        if not is_tracking:
            lab, color, ind = self.store_side(frame)
            if lab is not None:
                self.get_pair(num_object, lab, color, ind)
            self.milstone_flag = self.store(is_milestone=True)
            
        # self.memory.append(self.predict)
        #print(len(self.memory.list))

    
    def store_side(self, frame):
        img = frame.copy()
        point, center = hand_tracking(img).get_result()
        if point and len(self.boxls) > 0:
            red_center = side_finder(img, color='red')
            blue_center = side_finder(img, color='blue')
            tape = red_center + blue_center
            length_ls = []
            for (x, y) in tape:
                length_ls.append((self.get_k_dis((point[0], point[1]), (center[0], center[1]), (x, y)), (x, y)))
            x,y = min(length_ls, key=lambda x: x[0])[1]
            cv2.circle(img, (x,y), 10, [255, 255, 0], -1)
            ind = test_insdie((x, y), self.boxls)
            color = None
            if (x, y) in red_center:
                color = 'red'
            elif (x, y) in blue_center:
                color = 'blue'
            return self.predict[ind][1], color, ind
        else:
            return None, None, None
        cv2.imshow("point", img)
            


    def get_pair(self, num_object, label, color, index):
        '''
        pointing from left to right
        '''
        if self.once and self.once_lock and num_object == 2:
            if index == 0:
                self.memory.append(self.predict[0][1])
                if self.memory.full:
                    self.new_come_id = max(set(self.memory.list), key=self.memory.list.count)
                    # if self.new_come_id == label:
                    self.new_come_side = color
                    # else:
                    #     self.memory.clear()                                
                
            if self.memory.full and index == 1:    
                self.memory1.append(self.predict[-1][1])
                if self.memory1.full:
                    self.old_come_id = max(set(self.memory1.list), key=self.memory1.list.count)
                    # if self.old_come_id == label:
                    self.old_come_side = color
                    # else:
                    #     self.memory1.clear()
                    
            if self.memory.full and self.memory1.full:
                self.once_lock = False
                self.memory.clear()
                self.memory1.clear()
                print("new_come_id:{}, old_come_id:{}".format(self.new_come_id, self.old_come_id))
                print("new_come_side:{}, old_come_side:{}".format(self.new_come_side, self.old_come_side))
        
        
        '''
        pointing from left to right
        '''
        if not self.once and num_object == 2 and self.new_coming_lock:
            if index == 0:
                self.old_come_side = color
            elif index == 1:               
                self.memory.append(self.predict[1][1])
                if self.memory.full:
                    self.new_come_id = max(set(self.memory.list), key=self.memory.list.count)                    
                    self.new_come_side = color
                    self.memory.clear()

            if self.new_come_side and self.old_come_side:
                self.new_coming_lock = False
                print("new_come_id:{}".format(self.new_come_id))
                print("new_come_side:{}, old_come_side:{}".format(self.new_come_side, self.old_come_side))


        # if self.once and num_object == 2 and self.once_lock and self.predict[0][1] != self.predict[1][1]:
        #     self.memory.append(self.predict[0][1])
        #     self.memory1.append(self.predict[1][1])
        #     if self.memory.full:
        #         self.new_come_id = max(set(self.memory.list), key=self.memory.list.count)
        #         if self.new_come_id == label:
        #             self.new_come_side = color                                
        #     if self.memory1.full:
        #         self.old_come_id = max(set(self.memory1.list), key=self.memory1.list.count)
        #         if self.old_come_id == label:
        #             self.old_come_side = color
                
        #     if self.memory.full and self.memory1.full:
        #         self.once_lock = False
        #         self.memory.clear()
        #         self.memory1.clear()
        #         print("new_come_id:{}, old_come_id:{}".format(self.new_come_id, self.old_come_id))
        #         print("new_come_side:{}, old_come_side:{}".format(self.new_come_side, self.old_come_side))
        
        # if not self.once and num_object == 2 and self.new_coming_lock:
        #     if label < 10:
        #         self.memory.append(self.predict[-1][1])
        #         if self.memory.full:
        #             self.new_come_id = max(set(self.memory.list), key=self.memory.list.count)                    
        #             self.new_come_side = color
        #             self.memory.clear()
        #     elif label > 10:
        #         self.old_come_side = color
        #     if self.new_come_side and self.old_come_side:
        #         self.new_coming_lock = False
        #         print(self.new_come_side, self.old_come_side)     





    def warp(self, img):
        #pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
        pts1 = np.float32([[101,160],[531,133],[0,480],[640,480]])
        pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(640,480))
        return dst
            

    @staticmethod
    def get_k_dis((x1, y1), (x2, y2), (x, y)):
        coord = ((x, y), (x1, y1), (x2, y2))
        return Polygon(coord).area / distant((x1, y1), (x2, y2))




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

    @staticmethod
    def fromcv2tk(cv2image, master):
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        master.imgtk = imgtk
        master.configure(image=imgtk)
    
    def __del__(self):
        self.cap.release()
        


def main():
    
    start_time = time.time()
    detector = object_detector(start_time)
   
    while 1:
        detector.update(save=True)
        if cv2.waitKey(5) & 0xFF == 27:
            break
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
