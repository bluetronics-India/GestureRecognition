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

def recog(hand):
    """
    This function recognizes the hand gesture in the provided hand image

    It calculates the Euclidean Distance between the test image 'img'
    and all the sample images and then returns the label of the sample image with
    which the minimum euclidean distance is found

    The sample data should always be put in 'samples' folder
    """
    labDis = []
    """
    to use edge samples change the parameter of os.walk to '../data/edge_sample/'
    """
    for root1, dir1, files1 in os.walk("../data/samples/"):
        for file1 in files1:
            lab = file1.split('_')
            img = cv2.imread(os.path.join(root1, file1),cv2.IMREAD_GRAYSCALE)
            if len(hand.shape) == 3:
                c = cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)
            else:
                c = hand
            temp = img-c
            temp = np.power(temp,2)
            temp = np.sum(temp)
            euc = np.sqrt(temp)
            labDis.append([euc , lab[0]])
    arr = np.zeros((len(labDis)),np.float32)
    for i in range(len(labDis)):
        arr[i] = labDis[i][0]
    if arr.shape[0] == 0:
        return 'Error' 
    else:
        m = np.argmin(arr)
        label = labDis[m][1]
        return label

n=1

if len(sys.argv) == 1:
    print "---------------ERROR-------------------\nMissing an argument\n"
    print "Usage: python prjCS512.py <path_to_test_images_folder>"
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
        cv2.waitKey(10)
        '''
        Uncommenting the below line will allow using edge samples
        '''
        hand = cv2.Canny(hand,50,180)
        print (file, recog(hand))
print "Press any key to terminate"
cv2.waitKey(0)

"""
Output for current test data:

('a2.jpg', 'ClosedPalm')
('a3.jpg', 'ClosedPalm')
('a4.jpg', 'ClosedPalm')
('a5.jpg', 'ClosedPalm')
('a6.jpg', 'ClosedPalm')
('a7.jpg', 'OpenPalm')
('a8.jpg', 'OpenPalm')
('a9.jpg', 'OpenPalm')
('b2.jpg', 'ClosedPalm')
('b3.jpg', 'ClosedPalm')
('b4.jpg', 'ClosedPalm')
('b5.jpg', 'ClosedPalm')
('b6.jpg', 'ClosedPalm')
('b7.jpg', 'ClosedPalm')
('b8.jpg', 'ClosedPalm')
('b9.jpg', 'ClosedPalm')
('c2.jpg', 'one')
('c3.jpg', 'one')
('c4.jpg', 'one')
('c5.jpg', 'one')
('c6.jpg', 'one')
('c7.jpg', 'one')
('c8.jpg', 'one')
('c9.jpg', 'one')
('d2.jpg', 'two')
('d3.jpg', 'two')
('d4.jpg', 'two')
('d5.jpg', 'two')
('d6.jpg', 'two')
('d7.jpg', 'two')
('d8.jpg', 'two')
('d9.jpg', 'two')
('e2.jpg', 'three')
('e3.jpg', 'three')
('e4.jpg', 'three')
('e5.jpg', 'three')
('e6.jpg', 'three')
('e7.jpg', 'three')
('e8.jpg', 'three')
('e9.jpg', 'three')
('f2.jpg', 'four')
('f3.jpg', 'four')
('f4.jpg', 'four')
('f5.jpg', 'four')
('f6.jpg', 'four')
('f7.jpg', 'four')
('f8.jpg', 'four')
('f9.jpg', 'four')
('g2.jpg', 'up')
('g3.jpg', 'up')
('g4.jpg', 'up')
('g5.jpg', 'up')
('g6.jpg', 'up')
('g7.jpg', 'up')
('g8.jpg', 'up')
('g9.jpg', 'up')
('h2.jpg', 'down')
('h3.jpg', 'down')
('h4.jpg', 'down')
('h5.jpg', 'down')
('h6.jpg', 'down')
('h7.jpg', 'down')
('h8.jpg', 'down')
('h9.jpg', 'down')
"""

"""
Outputs for edge_sample:

('a2.jpg', 'up')
('a3.jpg', 'up')
('a4.jpg', 'up')
('a5.jpg', 'up')
('a6.jpg', 'up')
('a7.jpg', 'up')
('a8.jpg', 'one')
('a9.jpg', 'up')
('b2.jpg', 'one')
('b3.jpg', 'up')
('b4.jpg', 'up')
('b5.jpg', 'up')
('b6.jpg', 'up')
('b7.jpg', 'up')
('b8.jpg', 'up')
('b9.jpg', 'up')
('c2.jpg', 'one')
('c3.jpg', 'up')
('c4.jpg', 'up')
('c5.jpg', 'up')
('c6.jpg', 'up')
('c7.jpg', 'up')
('c8.jpg', 'one')
('c9.jpg', 'one')
('d2.jpg', 'one')
('d3.jpg', 'one')
('d4.jpg', 'one')
('d5.jpg', 'one')
('d6.jpg', 'one')
('d7.jpg', 'one')
('d8.jpg', 'one')
('d9.jpg', 'two')
('e2.jpg', 'one')
('e3.jpg', 'one')
('e4.jpg', 'one')
('e5.jpg', 'one')
('e6.jpg', 'one')
('e7.jpg', 'one')
('e8.jpg', 'one')
('e9.jpg', 'one')
('f2.jpg', 'up')
('f3.jpg', 'up')
('f4.jpg', 'up')
('f5.jpg', 'up')
('f6.jpg', 'up')
('f7.jpg', 'up')
('f8.jpg', 'up')
('f9.jpg', 'up')
('g2.jpg', 'one')
('g3.jpg', 'one')
('g4.jpg', 'up')
('g5.jpg', 'one')
('g6.jpg', 'one')
('g7.jpg', 'one')
('g8.jpg', 'up')
('g9.jpg', 'up')
('h2.jpg', 'one')
('h3.jpg', 'one')
('h4.jpg', 'one')
('h5.jpg', 'one')
('h6.jpg', 'one')
('h7.jpg', 'one')
('h8.jpg', 'one')
('h9.jpg', 'one')
"""