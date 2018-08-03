from __future__ import print_function
import cv2
import numpy as np
import sys
import math



def cartoon(input_image, a=14, N=3, p=43):
    input_image = cv2.imread(input_image)
    hsv = cv2.cvtColor(input_image,cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([63,101,61]), np.array([86,255,255]))
    green_mask = cv2.dilate(green_mask, kernel = np.ones((3,3),np.uint8))
    #green_mask = 255 - green_mask
    res = cv2.bitwise_and(input_image, input_image, mask=green_mask)
    input_image = cv2.subtract(input_image, res)
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

def padding(img, height, width):
    if img.shape[0] < height or img.shape[1] < width:
        offset = (((height-img.shape[0])//2, height-img.shape[0]-
        (height-img.shape[0])//2), ((width-img.shape[1])//2, width-img.shape[1]-
        (width-img.shape[1])//2), (0, 0))
        resize = np.pad(img, offset, 'constant')
        resize = add_background(resize)
        if resize.shape == (height, width):
            raise Exception('sorry i am fool')
        return resize
    else:
        return img



def concat_imgs(images):
    image_list = list(map(cartoon, images))
    height, width = 300, 200
    for i, image in enumerate(image_list):
        if i == 0:
            img1 = image
        else:
            img1 = np.concatenate((padding(img1, *(height, width)), padding(image, *(height, width))), axis=1)
    return img1


def add_background(image):
    background = cv2.imread('background.jpg')
    background = background[:image.shape[0], :image.shape[1]]
    image[image == 0] = background[image == 0]
    return image



def main():
    file = open("datasets/node.txt", "r")
    for lines in file:
        id1, id2, side1, side2 = int(lines[1]), int(lines[4]), str(lines[8:11]), str(lines[15:19])
        print(id1, id2, side1, side2)
    canvas = concat_imgs(['1.jpg', '2.jpg', '5.jpg'])
    canvas[canvas == 0] = 255
    cv2.imshow('ddd', canvas)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()