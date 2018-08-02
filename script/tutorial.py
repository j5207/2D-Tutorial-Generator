from __future__ import print_function
import cv2
import numpy as np
import sys
import math



def cartoon(input_image, a, N, p):
    input_image = cv2.imread(input_image)
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
    
    cv2.imshow('Toonified Image',finalimg)       
    cv2.waitKey(0)  


def main():
    file = open("datasets/node.txt", "r")
    for lines in file:
        print(lines[0])
    cartoon('2.jpg',14, 3, 43)

main()