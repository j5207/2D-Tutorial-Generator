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


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(320, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         print(x.shape)
#         x = x.view(-1,320)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.softmax(x, dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
        print(x.size())
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
        img1 = self.transform(img1)
        return (img1, result)
    def __len__(self):
        return len(self.reader)

class object_detector(): 
    def __init__(self, start): 		
        self.start_time = start
        self.stored_flag = False
        self.trained_flag = False
        self.boxls = None
        self.count = 1
        self.path = "/home/intuitivecompting/Desktop/color/Smart-Projector/script/datasets/"
        self.file = open(self.path + "read.txt", "w")

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

        draw_img = warp.copy()
        self.train_img = warp.copy()
        #-------------get the bounding box--------------
        self.get_minRect(draw_img, thresh, only=False, visualization=True)
        #--------------get bags of words and training-------#
        if not self.stored_flag:
            self.stored_flag = self.store(2.0)
        if self.stored_flag and not self.trained_flag: 
            self.trained_flag = self.train()
        if self.trained_flag: 
            self.train(is_train=False)

    def get_minRect(self, img, mask, only=True, visualization=True):
        (_,contours,_)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            for _, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area>600 and area < 10000:					
                    M = cv2.moments(contour)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    rect = cv2.minAreaRect(contour)
                    box = np.int0(cv2.boxPoints(rect))
                    x,y,w,h = cv2.boundingRect(contour)
                    self.boxls.append([x, y, w, h])
            #---------------sorting the list according to the x coordinate of each item
            if len(self.boxls) > 0:
                boxls_arr = np.array(self.boxls)
                self.boxls = boxls_arr[boxls_arr[:, 0].argsort()].tolist()
            for i in range(len(self.boxls)): 
                if visualization: 
                    x,y,w,h = self.boxls[i]
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    cv2.putText(img,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
        cv2.imshow('img', img)


    def warp(self, img):
        pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
        pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(640,480))
        return dst
    
    def store(self, store_time):
        num_object = len(self.boxls)
        frame = self.train_img
        #-------------capturing img for each of item--------------#
        for i in range(num_object):
            x,y,w,h = self.boxls[i]
            temp = frame[y:y+h, x:x+w, :]
            #class_folder_dir = self.path + str(i) + 'id'
            img_dir = os.path.join(self.path + "image", str(self.count) + ".jpg")
            #self.createFolder(class_folder_dir)
            cv2.imwrite(img_dir, temp)
            self.count += 1 
            self.file.write(img_dir + " " + str(i) + "\n") 
        if time.time() - self.start_time < store_time:
            print('output imgs ' + str(self.count) + 'img' )
            return False
        else:
            self.file.close()
            print('finish output')
            return True
    
    def train(self, is_train=True):
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        reader_train = self.reader(self.path, "read.txt")
        trainset = CovnetDataset(reader=reader_train, transforms=transforms.Compose([transforms.Scale((28, 28)),
                                                                                            transforms.ToTensor()
                                                                                       ]))
        trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=2)
        #cuda0 = torch.device('cuda:0')
        if is_train:
            for epoch in range(200):  # loop over the dataset multiple times
                running_loss = 0.0
                i = 0
                for data in trainloader:
                    # get the inputs
                    inputs, labels = data
                    inputs, labels = inputs, labels
                    # print(inputs)
                    inputs, labels = Variable(inputs), Variable(labels.long())
                    # print(inputs)
                    # print(labels)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = F.cross_entropy(outputs, labels.view(1, -1)[0])
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 20 == 19:    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / 20))
                        running_loss = 0.0
                    i += 1

            print('Finished Training')
            torch.save(net.state_dict(), f=self.path + 'model')
            return True
        else:
            net.load_state_dict(torch.load(f=self.path + 'model'))
            num_object = len(self.boxls)
            frame = self.train_img
            preprocess = transforms.Compose([transforms.Scale((200, 100)),
                                                        transforms.ToTensor()])
            for i in range(num_object):
                x,y,w,h = self.boxls[i]
                temp = frame[y:y+h, x:x+w, :]
                temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
                image = Image.fromarray(temp)
                img_tensor = preprocess(image)
                img_tensor.unsqueeze_(0)
                img_variable = Variable(img_tensor)
                out = np.argmax(net(img_variable).data.numpy()[0])
                print("{}th, out:{}".format(i, out))
            

        



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
    # object_detector(cap, save=True)
    start_time = time.time()
    detector = object_detector(start_time)
    while(1):
        detector.update(cap, save=True)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()