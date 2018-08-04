from __future__ import print_function
import cv2
import numpy as np
import sys
import math
from constant import *
import pickle
import glob


def cartoon(input_image, a=14, N=3, p=43):
    #input_image = cv2.imread(input_image)
    hsv = cv2.cvtColor(input_image,cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv,Green_low, Green_high)
    green_mask = cv2.dilate(green_mask, kernel = np.ones((3,3),np.uint8))

    hand_mask = cv2.inRange(hsv, Hand_low, Hand_high)
    hand_mask = cv2.dilate(hand_mask, kernel = np.ones((7,7),np.uint8))
    #green_mask = 255 - green_mask
    res = cv2.bitwise_and(input_image, input_image, mask=green_mask)
    res1 = cv2.bitwise_and(input_image, input_image, mask=hand_mask)
    input_image = cv2.subtract(input_image, res)
    input_image = cv2.subtract(input_image, res1)
    for _ in range(0,N):
        bilateral_filtimg = cv2.bilateralFilter(input_image,9,75,75)

    median_filtimg = cv2.medianBlur(bilateral_filtimg,5)

    [rows,cols,_] = median_filtimg.shape
    colorquantimg = median_filtimg
    for i in range(0,rows):
        for j in range(0,cols):
            xb = median_filtimg.item(i,j,0)
            xg = median_filtimg.item(i,j,1)
            xr = median_filtimg.item(i,j,2)  
            xb = math.floor(xb/a)*a 
            xg = math.floor(xg/a)*a
            xr = math.floor(xr/a)*a
            colorquantimg.itemset((i,j,0),xb)
            colorquantimg.itemset((i,j,1),xg)
            colorquantimg.itemset((i,j,2),xr)

    median_filtimg2 = cv2.medianBlur(input_image,5)

    edges = cv2.Canny(median_filtimg2,p,2*p)
    dialateimg =  cv2.dilate(edges,np.ones((3,3),'uint8'))
    edges_inv = cv2.bitwise_not(dialateimg)
    _,thresh = cv2.threshold(edges_inv,127,255,0)
    _,contours, _ = cv2.findContours(thresh,1,2)
    img_contours = cv2.drawContours(thresh, contours, -1, (0,0,0), 1)

    finalimg = colorquantimg.copy()
    for i in range(0,rows):
        for j in range(0,cols):
            if edges_inv.item(i,j) == 0:
                finalimg.itemset((i,j,0),0)
                finalimg.itemset((i,j,1),0)
                finalimg.itemset((i,j,2),0)

    return finalimg
    cv2.imshow('Toonified Image',finalimg)       
    cv2.waitKey(0)  

def padding(img, size, point):
    height, width = size
    cx, cy = point
    if img.shape[0] < height or img.shape[1] < width:
        y_minus = (height-img.shape[0])//2
        y_add = height-img.shape[0] - (height-img.shape[0])//2
        x_minus = (width-img.shape[1])//2
        x_add = width-img.shape[1] - (width-img.shape[1])//2
        offset = ((y_minus, y_add), (x_minus, x_add), (0, 0))
        resize = np.pad(img, offset, 'constant')
        # resize = add_background(resize)
        if resize.shape == (height, width):
            raise Exception('sorry i am fool')
        return resize, (cx+x_minus, cy+y_minus)
    else:
        return img, (cx, cy)



def concat_imgs(images, point_ls, visualization=False):
    if point_ls is not None:
        image_list = list(map(cartoon, images))
        height, width = 350, 350
        center_list = []
        for i, image in enumerate(image_list):
            if i == 0:
                img1, (x,y) = padding(image, (height, width), point_ls[i])
                center_list.append((x, y))
            else:
                img1 = padding(img1, (height, width), (0, 0))[0]
                image, (x, y) = padding(image, (height, width), point_ls[i])
                center_list.append((x+img1.shape[1], y))
                img1 = np.concatenate((img1, image), axis=1)
        if visualization:
            for point in center_list:
                cv2.circle(img1, point, 5, (255, 0, 0), -1)
        return img1, center_list
    else:
        image_list = list(map(cartoon, images))
        height, width = 350, 350
        for i, image in enumerate(image_list):
            if i == 0:
                img1 = padding(image, (height, width), (0, 0))[0]
            else:
                img1 = padding(img1, (height, width), (0, 0))[0]
                image = padding(image, (height, width), (0,0))[0]
                img1 = np.concatenate((img1, image), axis=1)
        return img1, None


def add_background(image):
    background = cv2.imread('test_imgs/background.jpg')
    background = background[:image.shape[0], :image.shape[1]]
    image[image == 0] = background[image == 0]
    return image

def get_center(img, visualization=False):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    object_mask = 255 - cv2.inRange(hsv, Green_low, Green_high)
    (_,object_contours, object_hierarchy)=cv2.findContours(object_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for i , contour in enumerate(object_contours):
        area = cv2.contourArea(contour)
        if object_hierarchy[0, i, 3] == -1 and area > max_area:	
            max_area = area				
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
    if visualization:
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
        cv2.imshow('haha', img)
    return (cx, cy)

def draw_arrow(img, pt1, pt2):
    alpha = 0.5
    pink = (255, 192, 203)
    output = img.copy()
    overlay = img.copy()
    pt1 = (pt1[0]+20, pt1[1])
    pt2 = (pt2[0]-20, pt2[1])
    cv2.arrowedLine(overlay, pt1, pt2, pink, 5)
    cv2.arrowedLine(overlay, pt2, pt1, pink, 5)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return img

class comic_book():
    num_instance = 0
    canvas = None
    def __init__(self, image_list, point=None):
        if comic_book.num_instance == 0:
            canvas= concat_imgs(image_list, point)[0]
            print(canvas.shape)
            canvas = add_background(canvas)
            cv2.putText(canvas,  "Step " + str(comic_book.num_instance), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
            comic_book.canvas = canvas
        else:
            canvas, center_list = concat_imgs(image_list, point)
            if point is not None:
                #for cx, cy in center_list:
                    #cv2.circle(canvas, (cx, cy), 5, (0, 255, 255), 1)
                draw_arrow(canvas, center_list[0], center_list[1])
            canvas = padding(canvas, (canvas.shape[0], comic_book.canvas.shape[1]), (0, 0))[0]
            canvas = add_background(canvas)
            cv2.putText(canvas,  "Step " + str(comic_book.num_instance), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
            canvas = np.concatenate((comic_book.canvas, canvas), axis=0)
            comic_book.canvas = canvas
        #cv2.imshow('img', comic_book.canvas)
        comic_book.num_instance += 1

def get_filename(ls, target):
    return list(filter(lambda x: int(str(x)[-5]) == target, ls))[0]


def main():
    data = pickle.load( open( "node.p", "rb" ))
    file_list = glob.glob('test_imgs/folder/*.jpg')
    item_list =  list(filter(lambda x: int(str(x)[-5]) < 5,  file_list))
    item_list = list(map(cv2.imread, item_list))
    comic_book(item_list)
    for predict, coord, outcome in data:
        img1 = cv2.imread(get_filename(file_list, predict[0]))
        img2 = cv2.imread(get_filename(file_list, predict[1]))
        out = cv2.imread(get_filename(file_list,outcome))
        comic_book([img1, img2], coord)
        comic_book([out])
    cv2.imwrite('img.jpg', comic_book.canvas)

    # canvas, center_list = concat_imgs(['1.jpg', '2.jpg', '5.jpg'])
    # draw_arrow(canvas, center_list[0], center_list[1])
    # canvas = add_background(canvas)
    # cv2.imshow('ddd', canvas)

if __name__ == '__main__':
    main()