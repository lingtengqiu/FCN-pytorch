import os 
import numpy as np
import torch
import torch.nn as nn
import vgg16


class FCN32s(vgg16.Vgg16_FCN):
    def __init__(self,num_class,pre_train =False):
        super(FCN32s,self).__init__(pre_train)
        self.score_fr = nn.Conv2d(4096,num_class,1)
        self.upscore = nn.ConvTranspose2d(num_class,num_class,64,stride =32,bias = False)
        self._init_weights()
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

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        
        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        #1/32
        h = self.score_fr(h)
        h = self.upscore(h)
        print h.shape
        return h[:,:,19:19+x.size()[2],19:19+x.size()[3]]

if __name__ =="__main__":
    fcn_net = FCN32s(21,True)
    for i in fcn_net.parameters():
        print type(i)
