import csv
import os
import glob

def get_all_test():

    files =[i.split('/')[-1] for i in list(glob.glob(os.path.join('/home/amrita95/PycharmProjects/darkflow2/test','*.jpg')))]
    
    bundle =[]
    csv_fname = os.path.join('/home/amrita95/PycharmProjects/darkflow2/udacity.csv')
    
    with open(csv_fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )
        a =[]
        for row in spamreader:
            img_name = row[0]
            if img_name not in files: 
                continue
            labels = row[1:]
            all = list()
            for i in range(0, len(labels), 5):
                xmin = int(labels[i])
                ymin = int(labels[i + 1])
                xmax = int(labels[i + 2])
                ymax = int(labels[i + 3])
                class_idx = int(labels[i + 4])
                all += [[class_idx,xmin, ymin, xmax, ymax]]
            bundle += [[img_name,  all]]

    return bundle
    
def cal_iou(list1,list2):
    a = intersect(list1,list2)
    return a/union(list1,list2)

def intersect(list1,list2):
    x1,x2,x3,x4 = list1[0],list1[2],list2[0],list2[2]
    y1,y2,y3,y4 = list1[1],list1[3],list2[1],list2[3]

    w = overlap(x1,x2,x3,x4)
    h = overlap(y1,y2,y3,y4)
    if w<0 or h<0:
        return 0
    area =w*h

    return area

def overlap(x1,x2,x3,x4):
    left = max(x1,x3)
    right = min(x2,x4)
    return right - left

def union(list1,list2):
    i = intersect(list1,list2)
    union = (list1[2]-list1[0])*(list1[3]-list1[1]) + (list2[2]-list2[0])*(list2[3]-list2[1]) - i
    return union

def new_KPI(tests_bundle,pred_bundle):
    print(len(tests_bundle),len(pred_bundle))
    for img in tests_bundle:
        precision=0
        recall =0
        if pred_bundle[0] == img[0]:
            total = len(img[1])
            proposals = len(pred_bundle[1])
            avg_iou =0
            correct=0

            for pred in pred_bundle[1]:
                best_iou = 0

                for true in img[1]:
                    iou= cal_iou(true[1:],pred[1:])
                    if (iou > best_iou):
                        best_iou = iou

                if(best_iou>0.30):
                    correct += 1

                avg_iou+=best_iou


            if proposals:
                precision = correct/proposals
            if total:
                recall= correct/total
                avg_iou = avg_iou/total

            return img[0],total,proposals,correct,avg_iou,recall,precision
