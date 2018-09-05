#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-30 09:43
# * Last modified : 2018-08-30 09:43
# * Filename      : loss_funcition.py
# * Description   : 
# **********************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum().float()
    return loss
class pix_cross_entropy2d(nn.Module):
    def __call__(self,inputs,labels,weight = None):
        '''
        very import part for here
        inputs : n,c,h,w
        labels : n,h,w
        use to 2-dim cross_entropy
        '''
        n,c,h,w = inputs.shape
        log_p = F.log_softmax(inputs,dim=1)
        #n,h,w,c
        log_p = log_p.transpose(1,2).transpose(2,3).contiguous()
        log_p = log_p[labels.view(n,h,w,1).repeat(1,1,1,c)>=0]
        #in here we change the channel so the view is according to channels save
        log_p = log_p.view(-1,c)

        #labels
        mask = labels>=0
        labels = labels[mask]
        loss = F.nll_loss(log_p,labels,weight=weight,size_average = False)
        loss = loss/mask.data.sum().float()
        return loss
        

