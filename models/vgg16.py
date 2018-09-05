import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
import architeture
class Vgg16_FCN(nn.Module):
    def __init__(self,pre_train = False):
        self.pre_train = pre_train
        super(Vgg16_FCN,self).__init__()
        #conv1
        self.conv1_1 = nn.Conv2d(3,64,3,padding =100)
        self.relu1_1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        self.relu1_2 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2,2)
        #conv2
        self.conv2_1 =  nn.Conv2d(64,128,3,padding =1)
        self.relu2_1 = nn.ReLU(True)
        self.conv2_2 = nn.Conv2d(128,128,3,padding =1)
        self.relu2_2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2,2)

        #conv3
        self.conv3_1 = nn.Conv2d(128,256,3,padding =1)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(256,256,3,padding =1)
        self.relu3_2 = nn.ReLU(True)
        self.conv3_3 = nn.Conv2d(256,256,3,padding =1)
        self.relu3_3 = nn.ReLU(True)
        #1/8 
        self.pool3 = nn.MaxPool2d(2,2)
        #conv4
        self.conv4_1 = nn.Conv2d(256,512,3,padding =1)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(512,512,3,padding =1)
        self.relu4_2 = nn.ReLU(True)
        self.conv4_3 = nn.Conv2d(512,512,3,padding =1)
        self.relu4_3 = nn.ReLU(True)
        #1/16
        self.pool4 = nn.MaxPool2d(2,2)
        #conv5
        self.conv5_1 = nn.Conv2d(512,512,3,padding =1)
        self.relu5_1 = nn.ReLU(True)
        self.conv5_2 = nn.Conv2d(512,512,3,padding =1)
        self.relu5_2 = nn.ReLU(True)
        self.conv5_3 = nn.Conv2d(512,512,3,padding =1)
        self.relu5_3 = nn.ReLU(True)
        #1/32
        self.pool5 = nn.MaxPool2d(2,2)
        #fc6
        #here we use full-connect
        self.fc6 = nn.Conv2d(512,4096,7)
        self.relu6 = nn.ReLU(True)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096,4096,1)
        self.relu7 = nn.ReLU(True)
        self.drop7 = nn.Dropout2d()

    def _init_weights(self):
        #init the weight
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            #transpose conv init
            if isinstance(m,nn.ConvTranspose2d):
                initial_weight = self.get_upsampling_weight(m.in_channels,m.out_channels,m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
        if self.pre_train:
            print "get_vgg16 para"
            vgg16 = vgg.vgg16(pretrained  = True)
            self.copy_paras_from_vgg16(vgg16)
    #this part to get the vgg16 para
    def copy_paras_from_vgg16(self,vgg16):
        features =[
                self.conv1_1,self.relu1_1,
                self.conv1_2,self.relu1_2,
                self.pool1,
                self.conv2_1,self.relu2_1,
                self.conv2_2,self.relu2_2,
                self.pool2,
                self.conv3_1,self.relu3_1,
                self.conv3_2,self.relu3_2,
                self.conv3_3,self.relu3_3,
                self.pool3,
                self.conv4_1,self.relu4_1,
                self.conv4_2,self.relu4_2,
                self.conv4_3,self.relu4_3,
                self.pool4,
                self.conv5_1,self.relu5_1,
                self.conv5_2,self.relu5_2,
                self.conv5_3,self.relu5_3,
                self.pool5,
                ]
        ##copy
        #copy conv layer
        print "get pre_train para from vgg16"
        for l1,l2 in zip(vgg16.features,features):
            if isinstance(l1,nn.Conv2d) and isinstance(l2,nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i,name in zip([0,3],['fc6','fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self,name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
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
	return h
    def get_upsampling_weight(self,in_channels,out_channels,kernel_size):
        factor = (kernel_size+1)//2
        if kernel_size %2 == 1:
            center = factor - 1
        else:
            center = factor -0.5
        og = np.ogrid[:kernel_size,:kernel_size]

        filt = (1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
        weight = np.zeros((in_channels,out_channels,kernel_size,kernel_size),dtype =np.float32)
        weight[range(in_channels),range(out_channels),:,:]= filt
        return torch.from_numpy(weight)

if __name__ =="__main__":

    net = Vgg16_FCN(True)
    x = torch.randn(1,3,224,223)
    out = net(x)
    g = architeture.make_dot(out,net.state_dict())
    g.view()
