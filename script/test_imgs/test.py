import numpy as np
import cv2
import os
import argparse
import data_utils
import shape_context
import morphology_utils
import multiprocessing
import matplotlib.pyplot as plt
import _pickle as pickle
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans

def reader(path, mode):
    path= ''.join([path, mode])
    f = open(path)
    data = f.readlines()
    image_ls = []
    label_ls = []
    for line in data:
        objects = line.split()
        image_ls.append(cv2.imread(objects[0]))
        label_ls.append(int(objects[1]))
    #     data_ls.append((objects[0], objects[1]))
    # lables = set(map(lambda x: int(x[1]), data_ls))
    # output = {}
    # for label in lables:
    #     filter_list = list(filter(lambda x: int(x[1]) == label, data_ls))
    #     map_list = list(map(lambda x: x[0], filter_list))
    #     output[label] = map_list
        
    return image_ls, label_ls

base_path = "/Users/jihan/Desktop/Smart-Projector/script/datasets/"
images, labels = reader(base_path, "read.txt")
sc = shape_context.ShapeContext()
def sc_array(img):
    sp = morphology_utils.shape_points(img, 30)
    bh = sc.compute(sp)
    return bh
num_cores = multiprocessing.cpu_count()
sc_arrays_list  = Parallel(n_jobs=num_cores)(delayed(sc_array)(img) for img in images)
sc_arrays = np.array(sc_arrays_list)
print(sc_array)
# img = cv2.imread('saved01.jpg')
# h = sc_array(img)
# print(h.shape)











# from __future__ import print_function
# import cv2
# import numpy as np
# from constant import *

# def warp_img(img):
#     #pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
#     pts1 = np.float32([[101,160],[531,133],[0,480],[640,480]])
#     pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
#     M = cv2.getPerspectiveTransform(pts1,pts2)
#     dst = cv2.warpPerspective(img,M,(640,480))
#     return dst

# def camrectify(frame):
#         mtx = np.array([
#             [509.428319, 0, 316.944024],
#             [0.000000, 508.141786, 251.243128],
#             [0.000000, 0.000000, 1.000000]
#         ])
#         dist = np.array([
#             0.052897, -0.155430, 0.005959, 0.002077, 0.000000
#         ])
#         return cv2.undistort(frame, mtx, dist)

# def get_crit(mask):
#     (_,contours, hierarchy)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     max_area = 0
#     crit = None
#     for i , contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if area > max_area and hierarchy[0, i, 3] == -1:
#             max_area = area
#             crit = area
#     return crit

# def get_objectmask(img):
#     hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     green_mask = cv2.inRange(hsv, Green_low, Green_high)
#     hand_mask = cv2.inRange(hsv, Hand_low, Hand_high)
#     hand_mask = cv2.dilate(hand_mask, kernel = np.ones((7,7),np.uint8))
#     skin_mask = cv2.inRange(hsv, Skin_low, Skin_high)
#     skin_mask = cv2.dilate(skin_mask, kernel = np.ones((7,7),np.uint8))
#     thresh = 255 - green_mask
#     thresh = cv2.subtract(thresh, hand_mask)
#     thresh = cv2.subtract(thresh, skin_mask)
#     thresh[477:, 50:610] = 0
#     return thresh


# cap = cv2.VideoCapture(0)
# while 1:
#     OK, origin = cap.read()
#     if OK:
#         rect = camrectify(origin)
#         warp = warp_img(rect)
#         thresh = get_objectmask(warp)

#         draw = warp.copy()

#         (_,contours, hierarchy)=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         max_area = 0
#         crit = None
#         for i , contour in enumerate(contours):
#             area = cv2.contourArea(contour)
#             if area > max_area and hierarchy[0, i, 3] == -1:
#                 max_area = area
#                 crit = contour
#         for x in crit:
#             cv2.circle(draw, (x[0][0], x[0][1]), 8, (0,0,255), 1.0)
#         cv2.imshow('dd', draw)

    #print(crit)