import os
import csv
from sklearn.cluster import KMeans,Birch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
dumps = list()

csv_fname = os.path.join('/home/amrita95/PycharmProjects/darkflow2/udacity.csv')
with open(csv_fname, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )
    for row in spamreader:
        img_name = row[0]
        w = 1920
        h = 1200

        labels = row[1:]
        all = list()
        for i in range(0, len(labels), 5):
            xmin = int(labels[i])
            ymin = int(labels[i + 1])
            xmax = int(labels[i + 2])
            ymax = int(labels[i + 3])
            class_idx = int(labels[i + 4])
            dumps.append([(xmax-xmin),(ymax-ymin)])

dumps = np.array(dumps)

plt.scatter((dumps[:,0]),(dumps[:,1]), label='True Position',s = 0.2)
kmeans = KMeans(n_clusters = 5).fit(dumps)

anchors= kmeans.cluster_centers_*13/416.0
'''
anchors =np.array([[1.08,1.19],[3.42,4.41],[6.63,11.38],[9.42,5.11], [16.62,10.52]])
anchors = anchors*416/13
stride = 32
stride_h = 10
stride_w = 3
colors = [(255,0,0),(255,255,0),(0,255,0),(0,0,255),(0,255,255),(55,0,0),(255,55,0),(0,55,0),(0,0,25),(0,255,55)]

[H,W] = (800,800)
blank_image = np.zeros((H,W,3),np.uint8)
for i in range(len(anchors)):
    (w,h) = anchors[i]
    print(w,h)
    cv2.rectangle(blank_image,(10+i*stride_w,10+i*stride_h),(int(w),int(h)),colors[i],thickness=3)

cv2.imshow('Image',blank_image)
cv2.imwrite('/home/amrita95/PycharmProjects/darkflow2/anchor.jpg',blank_image)
cv2.waitKey(10000)
'''
plt.show()







