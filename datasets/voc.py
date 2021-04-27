#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 
# * Create time   : 2018-08-29 18:44
# * Last modified : 2018-08-29 18:44
# * Filename      : voc.py
# * Description   : 
# **********************************************************

import collections
import os

import numpy as np
import scipy.io
import cv2
import torch
from torch.utils import data
from PIL import Image
class VOCClassSegBase(data.Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    #BGR
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    def __init__(self,root,train =True,transform = None):
        self.root = root 
        self.train = train
        self.transform = transform
        dataset_dir = os.path.join(self.root,"VOCdevkit/VOC2012")
        self.files = collections.defaultdict(list)

        for states in["train","val"]:
            img_file_name = os.path.join(
                dataset_dir,
                "ImageSets/Segmentation/{}.txt".format(states)
                    )
            with open(img_file_name,'r') as reader:
                for img_name_line in reader:
                    img_name_line = img_name_line.strip()
                    img_file =os.path.join(dataset_dir,"JPEGImages/{}.jpg".format(img_name_line))
                    label_file = os.path.join(dataset_dir,"SegmentationClass/{}.png".format(img_name_line))
                    self.files[states].append( {'img':img_file,'label':label_file})

    def __len__(self):
        if self.train:
            return len(self.files['train'])
        else:
            return len(self.files['val'])
    def __getitem__(self,index):
        if self.train:
            data_file = self.files['train'][index]
        else:
            data_file = self.files['val'][index]
        img_file = data_file['img']
        label_file = data_file['label']
        img = cv2.imread(img_file)
        #label for P mode
        label = Image.open(label_file).convert("P")
        img = np.asarray(img,np.float32)
        label = np.array(label,np.int32)
        img -=self.mean_bgr
        label[label == 255] = -1
        # img c,w,h
        img = torch.from_numpy(np.transpose(img,(2,0,1))).float()
        #label w,h,c  P mode
        label = torch.from_numpy(label).long()
        return img,label
class VOC2012ClassSeg(VOCClassSegBase):
    def __init__(self,root,train=True,transform =None):
        super(VOC2012ClassSeg,self).__init__(root,train,transform)
if __name__ =="__main__":
    voc = VOCClassSegBase(root = "./",train=False)
    for i in range(100):
        img,label = voc[i]
        cv2.imshow("fuck",img)
        cv2.imshow("label",label)
        cv2.waitKey()
