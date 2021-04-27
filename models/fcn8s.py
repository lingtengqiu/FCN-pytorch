#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 
# * Create time   : 2018-08-29 15:16
# * Last modified : 2018-08-29 15:16
# * Filename      : fcn8s.py
# * Description   : we have two part of here one is at once second is mutil stage 
# **********************************************************
import torch
import os
import torch.nn as nn
import torchvision.models.vgg as vgg
import vgg16
import architeture
class FCN8s(vgg16.Vgg16_FCN):
    def __init__(self,n_class,fcn16s = None,pre_train =False):
        super(FCN8s,self).__init__(pre_train) 
        self.score_fr = nn.Conv2d(4096,n_class,1)
        self.score_pool4 = nn.Conv2d(512,n_class,1)
        self.score_pool3 = nn.Conv2d(256,n_class,1)
        
        self.upscore2 = nn.ConvTranspose2d(n_class,n_class,4,stride = 2,bias = False)
        self.upscore8 = nn.ConvTranspose2d(n_class,n_class,16,stride = 8,bias = False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class,n_class,4,stride =2,bias=False)
      
    def _init_weights(self,fcn16s):
        if pre_model == None or self.pre_train == False:
            super(FCN8s,self)._init_weights()
        else:
            for name,l1  in fnc16s.name_childern():
                try:
                    l2 = getattr(self,name)
                    l2.weight
                except Exception:
                    continue
                assert l1.weight.size() == l2.weight.size()
                l2.weight.data.copy_(l1.weight.data)
                if l1.bias is not None:
                    assert l1.bias.data.size() == l2.bias.data.size()
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
        pool3 = h
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
        #1/32
        h = self.score_fr(h)
        #1/16
        up_fr = self.upscore2(h)

        h = self.score_pool4(pool4)
        h = h[:,:,5:5+up_fr.size()[2],5:5+up_fr.size()[3]]
        score_pool4 = h
        #1/8
        up_pool4 = self.upscore_pool4(score_pool4+up_fr)

        h = self.score_pool3(pool3)
        h = h[:,:,9:9+up_pool4.size()[2],9:9+up_pool4.size()[3]]
        
        score_pool3 = up_pool4+h
        h = self.upscore8(score_pool3)

        h = h[:,:,31:31+x.size()[2],31:,31+x.xize()[3]].contiguous()

class FCN8s_At_Once(vgg16.Vgg16_FCN):
    def __init__(self,num_class,pre_train=False):
        super(FCN8s_At_Once,self).__init__()
        self.score_fr  = nn.Conv2d(4096,num_class,1)
        self.score_pool4 = nn.Conv2d(512,num_class,1)
        self.score_pool3 = nn.Conv2d(256,num_class,1)

        self.up_fr = nn.ConvTranspose2d(num_class,num_class,4,stride =2,bias=False)
        self.up_pool4 = nn.ConvTranspose2d(num_class,num_class,4,stride =2,bias= False)
        self.up_pool3 = nn.ConvTranspose2d(num_class,num_class,16,stride =8,bias =False)
        self._initialize_weights()
    def _initialize_weights(self):
        print("begin to initial our para")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
    def forward(self,x):
        h = x
        print torch.sum(h)
        #conv1
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)
        #conv2
        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        #conv3
        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        #1/8
        pool3 = h
        #conv4
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        #1/16
        pool4  = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        #1/8
        h = self.pool5(h)
        #fc
        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        #speical here
        
        score_fc = self.score_fr(h)
        up_fc = self.up_fr(score_fc) 
        #in here to scale the pool4 score stablity
        score_pool4 = self.score_pool4(pool4 *0.01)
        score_pool4 = score_pool4[:,:,5:5+up_fc.shape[2],5:5+up_fc.shape[3]]
        #combind2
        com_pool4 = score_pool4+up_fc
        up_pool4 = self.up_pool4(com_pool4)

        score_pool3 = self.score_pool3(pool3*0.0001)
        score_pool3 = score_pool3[:,:,9:9+up_pool4.shape[2],9:9+up_pool4.shape[3]]
        com_pool3 = score_pool3+up_pool4

        h = self.up_pool3(com_pool3)
        h = h [:,:,31:31+x.shape[2],31:31+x.shape[3]]
        return h
    def copy_params_from_vgg16(self, vgg16):
        print "copy from vgg16"
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
def get_paramers(model,bias = False):
        modules_skipped =(
                nn.Dropout2d,
                nn.ReLU,
                nn.MaxPool2d,
                nn.Sequential,
                FCN8s_At_Once
                )
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight
            elif isinstance(m, nn.ConvTranspose2d):
                # weight is frozen because it is just a bilinear upsampling
                if bias:
                    assert m.bias is None
            elif isinstance(m, modules_skipped):
                continue
            else:
                raise ValueError('Unexpected module: %s' % str(m))
if __name__ == "__main__":
    fcn_at_once = FCN8s_At_Once(21,True)
    vgg_models  = vgg.vgg16(True) 
    fcn_at_once.copy_params_from_vgg16(vgg_models)
    x= torch.randn(1,3,224,224)
    out = fcn_at_once(x)
    for i in get_paramers(fcn_at_once):
        print i

    #g = architeture.make_dot(out,fcn_at_once.state_dict())
    #g.view()
