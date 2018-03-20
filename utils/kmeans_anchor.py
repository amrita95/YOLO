import os
import csv
from sklearn.cluster import KMeans

import pandas as pd

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


kmeans = KMeans(n_clusters = 5).fit(dumps)
print(kmeans.cluster_centers_,kmeans.cluster_centers_*13/416.0)







