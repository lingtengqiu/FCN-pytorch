import os.path as osp


import torch
import torchvision
import tool

def VGG16(url,pretrained=False):
    model = torchvision.models.vgg16(pretrained)
    state_dict = torch.load(url)
    model.load_state_dict(state_dict)
    return model


def _get_vgg16_pretrained_model():
    return tool.cached_download(
        url='http://drive.google.com/uc?id=0B9P1L--7Wd2vLTJZMXpIRkVVRFk',
        path=osp.expanduser('~/data/models/pytorch/vgg16_from_caffe.pth'),
        md5='aa75b158f4181e7f6230029eb96c1b13',
    )

if __name__ =="__main__":
    vgg16 = VGG16("./vgg16_from_caffe.pth",True) 
    vgg_pre = torchvision.models.vgg16(True)
    for i,j in zip(vgg16.parameters(),vgg_pre.parameters()):
        print torch.sum(i-j)

