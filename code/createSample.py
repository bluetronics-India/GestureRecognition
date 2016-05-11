"""
OpenCV version: 2.4.11
Python 2.7
"""
from pylab import *
import cv2
import numpy as np
import matplotlib as mp
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show
import os

n=1

if len(sys.argv) == 1:
    print "---------------ERROR-------------------\nMissing an argument\n"
    print "Usage: python createSample.py <path_to_original_sample_images_folder>"
    print "---------------------------------------"
    sys.exit(0)

testPath = sys.argv[1]

conFile = open('prj.config','r')
Thresh = conFile.readlines()[0].split()
Thresh = int(Thresh[0])

for root, dirs, files in os.walk(testPath):
    for file in files:
        string=os.path.join(root, file)
        src=cv2.imread(string)
        cv2.imshow("src",src)
        h,w = src.shape[:2]
        ratio = w/float(h)
        src=cv2.resize(src,(int(ratio*500),500))
        hsv=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
        hsvSplit=cv2.split(hsv)
        #Bluring the imgage to enhance the hand
        hsvSplit[1]=cv2.GaussianBlur(hsvSplit[1],(9,9),0)
        ret,hsvSplit[1] = cv2.threshold(hsvSplit[1], Thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy=cv2.findContours(hsvSplit[1],cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        hand=np.zeros(src.shape,src.dtype)

    ##initilizing variables for finding max contours
        max=0
        pt=-1
    ## Finding largest contour
        for i in range(0,len(contours),1):
            area=cv2.contourArea(contours[i])
            if area>max:
                max=area
                pt=i


        cv2.drawContours(hand, contours, pt, (255,255,255), cv2.cv.CV_FILLED)
        '''
        Uncommment the below line to get edge samples rather than binary
        '''
        #hand = cv2.Canny(hand,50,180)
        
        """
        The code segment below was used to create the sample data images from a given folder of images

        """       
        label = ''
        if 'a' in file:
            label = 'OpenPalm'
        elif 'b' in file:
            label = 'ClosedPalm'
        elif 'c' in file:
            label = 'one'
        elif 'd' in file:
            label = 'two'
        elif 'e' in file:
            label = 'three'
        elif 'f' in file:
            label = 'four'
        elif 'g' in file:
            label = 'up'
        elif 'h' in file:
            label = 'down'

        cv2.imwrite("../data/samples/"+label+'_'+str(n)+".jpg",hand)
        #cv2.waitKey(20)
        n+=1