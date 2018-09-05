#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-29 14:32
# * Last modified : 2018-08-29 14:32
# * Filename      : fcn16s.py
# * Description   : 
# **********************************************************
import torch
import torch.nn as nn
import os
import numpy as np
import vgg16
class FCN16s(vgg16.Vgg16_FCN):
    def __init__(self,num_class,fcn32s = None,pre_train = False):
        super(FCN16s,self).__init__(pre_train)
        self.score_fr = nn.Conv2d(4096,num_class,1)
        self.score_pool4 = nn.Conv2d(512,num_class,1)

        self.upscore2 = nn.ConvTranspose2d(num_class,num_class,4,stride =2,bias = False)
        self.upscore16 = nn.ConvTranspose2d(num_class,num_class,32,stride = 16,bias=False)
        self._init_weights(fcn32s)
    def _init_weights(self,fcn32s=None):
        if self.pre_train == False or fcn32s == None:
            super(FCN16s,self)._init_weights()
        else:
            for name,l1 in fcn32s.name_childern():
                #this part to relax the relu,drop,and no equall name
                try:
                    l2 = getattr(self,name)
                    l2.weight
                except Exception:
                    continue
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)
    def forward(self,x):
        h = x 
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h =self.pool1(h)
        
        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h
        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        #32
        h = self.score_fr(h)
        upscore2 = self.upscore2(h)
        print upscore2.shape
        h = self.score_pool4(pool4)
        h = h[:,:,5:5+upscore2.shape[2],5:5+upscore2.shape[3]]
        h +=upscore2
        h = self.upscore16(h)
        h = h[:,:,27:27+x.size()[2],27:27+x.size()[3]]
        
if __name__ == "__main__":
    fcn16s = FCN16s(16,pre_train =True)
    device = torch.device("cuda:0" if torch.cuda.is_available()else"cpu")
    fcn16s.to(device)
    x = torch.randn(1,3,100,100)
    x = x.to(device)
    fcn16s(x)

