#!/usr/bin/python
# -*- coding: UTF-8 -*-
# **********************************************************
# * Author        : lingteng qiu
# * Email         :
# * Create time   : 2018-08-16 16:40
# * Last modified : 2018-08-16 16:40
# * Filename      : main.py
# * Description   : this is a demo about main.py for my back project
# **********************************************************
import torch
import torchvision
import numpy as np
import fire
import config
import models
import utils
import os
import datasets
import time
import shutil
import utils
import cv2
import tqdm
from PIL import Image
from PIL import ImagePalette
import copy
import PIL
import ext_model
opts = config.DefaultConfig()
def train(**kwargs):
    '''
    para :
        opts:the para from your
    return:i
        the train model
    '''
    opts.parse_kwargs(**kwargs)
    torch.cuda.manual_seed(1337)
    viz = utils.visualizer.Visualizer()
    #net = getattr(models,opts.model)(21,True)
    net = ext_model.FCN8sAtOnce(21)
    pre_vgg = ext_model.VGG16("./ext_model/vgg16_from_caffe.pth",pretrained = True)
    net.copy_params_from_vgg16(pre_vgg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #step1 model
    net.to(device)
    #step2 data_set
    train_set = datasets.VOC2012ClassSeg("./datasets",train=True)
    val_set = datasets.VOC2012ClassSeg("./datasets",train=False)
    data_loader =  {"train":torch.utils.data.DataLoader(train_set,batch_size = 1,shuffle = True,num_workers = True,pin_memory =True),"val":torch.utils.data.DataLoader(val_set,batch_size = 1,shuffle = False)}
    #step3 criterion,optim
    optimer = torch.optim.SGD(
            [
                {'params':ext_model.get_parameters(net,False)},
                {'params':ext_model.get_parameters(net,True),'lr':2*opts.lr,'weight_decay':0}
                ],
            lr = opts.lr,weight_decay=opts.weight_decay,momentum =opts.momentum
        )
    criterion = models.cross_entropy2d
    sche_lr = torch.optim.lr_scheduler.MultiStepLR(optimer,[40,50],gamma = 0.1) 
    #step4 compare
    best_loss = 0.0
    since = time.time()
    #this part use to draw plot
    plot_lin_win = 'plot_val_win'
    max_epoch = int(np.ceil(1. * opts.iteration/len(data_loader['train'])))
    plot_line_train = 'plot_train_win'
    img_orign_win = None
    epoch = 0 
    class_name = data_loader["train"].dataset.class_names
    since = time.time()
    net.load_state_dict(torch.load("./check_point/FCN_0905_04:42:42.pt"))
    mean_iu_win = None
    #step5 training
    for _epoch in tqdm.trange(epoch,max_epoch,desc='train_epoch',ncols=90):
        net.train()
        epoch_loss=0.0
        sche_lr.step()
        for batch_idx,(data,target) in tqdm.tqdm(enumerate(data_loader['train']),total=len(data_loader['train']),desc = "Trainning epoch : {}".format(_epoch),ncols=80,leave = False):
            '''
            see the input and label
            '''
            idx = batch_idx + _epoch*len(data_loader['train'])
            #img_orign_win = viz.images(img_ = data,win = img_orign_win,title = 'original')
            data = data.to(device)
            target = target.to(device) 
            optimer.zero_grad()
            with torch.set_grad_enabled(True):
                scores = net(data)
                loss = criterion(scores,target)
                loss.backward()
                optimer.step()
            epoch_loss += loss.item()
            if idx  % opts.interval_validate == 0:
                #process validation for val
                val_loss,acc,acc_cls,mean_iu,fwavacc = val(net,data_loader['val'],viz,criterion,idx)
                x =  np.asarray([idx// opts.interval_validate])
                y = (np.asarray([val_loss]))
                ius = (np.asarray([mean_iu]))
                plot_lin_win = viz.plot(x,y,plot_lin_win,'val_Loss','loss')
                mean_iu_win =  viz.plot(x,ius,mean_iu_win,'mean_iu','iu')
                print "val_loss in here is {}".format(val_loss)
                if mean_iu>best_loss:
                    best_loss = mean_iu
                    best_model = net.state_dict()
                    prefix = './check_point/'+"FCN"+"_"
                    name = time.strftime(prefix +"%m%d_%H:%M:%S.pt")
                    torch.save(best_model,name)
                net.train()
        epoch_loss = epoch_loss/len(data_loader["train"])
        now =1.0*(time.time() - since)/60
        x = np.asarray([now])
        y = np.asarray([epoch_loss])
        plot_line_train = viz.plot(x,y,plot_line_train,'train_loss','loss') 

def val(net,val_loader,viz,criterion,iteration = None):
    '''
    validation our model is well?
    '''
    training = net.training
    #eval mode for drop 
    device =  torch.device("cuda:0")
    net.eval()
    val_true_win = 'val_true_image'
    label_name = val_loader.dataset.class_names
    val_loss =0.0
    label_trues = []
    label_preds = []
    for batch_idx,(data,target) in tqdm.tqdm(enumerate(val_loader,1),total = len(val_loader),desc='Validation iteration {}'.format(iteration),ncols=80,leave = False):
        data = data.to(device)
        target = target.to(device)
        scores = net(data)
        with torch.set_grad_enabled(False):
            loss =criterion(scores,target)
        val_loss +=loss.item()

        imgs =data.detach().cpu().numpy()
        #get the idx value for which labels?
        lbl_pred = scores.max(1)[1].cpu().numpy()[:,:,:]
        lbl_true  = target.detach().cpu().numpy()
        label_trues.append(lbl_true)
        label_preds.append(lbl_pred)
    acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(label_trues,label_preds,21)

    loss  = val_loss/len(val_loader) 
    net.train() 
    return loss,acc,acc_cls,mean_iu,fwavacc

def untransform(imgs,lbl):
    #this part to change back to orignal graph
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    imgs = np.transpose(imgs,(1,2,0))
    imgs +=mean_bgr
    imgs [imgs>255] = 255
    imgs =imgs.astype(np.uint8)
    imgs = np.ascontiguousarray(imgs[:,:,::-1])
    return imgs,lbl




def test(**kwargs):
    opts.parse_kwargs(**kwargs)
    print("test begin")
    test_loader = datasets.VOC2012ClassSeg("./datasets",train = False)
    test_loader = torch.utils.data.DataLoader(test_loader,batch_size = 1,shuffle =False)
    label_name = test_loader.dataset.class_names
    result = "./result"
    if os.path.exists(result):
        shutil.rmtree(result)
    os.mkdir(result)
    net = ext_model.fcn8s.FCN8sAtOnce(21)
    device = torch.device("cuda:0")
    net.load_state_dict(torch.load(opts.load_model_path))
    net.to(device)
    for batch_idx,(data,target) in tqdm.tqdm(enumerate(test_loader,1),total = len(test_loader),desc = "process :testing",ncols = 100):
        data = data.to(device)
        target = target.to(device)
        scores = net(data)
        #segmentation
        imgs = data.detach().cpu().numpy()
        lbl_pred = scores.max(1)[1].cpu().numpy()[:,:,:]
        lbl_true = target.detach().cpu().numpy()
        for img,lt,lp in zip(imgs,lbl_true,lbl_pred):
            img,lt = untransform(img,lt)
            viz_images = utils.visualize_segmentation(img = img,lbl_true= lt ,lbl_pred=lp,n_class =len(label_name),label_names = label_name)
            viz_images  = cv2.cvtColor(viz_images,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(result,"{}.jpg".format(batch_idx)),viz_images)




def help():
    '''
    print help imformation
    in here we have more ideal
    train_data_root
    test_data_root
    load_model_path
    bath_size
    use_gpu
    num_worker
    print_freq
    max_epoch
    lr
    lr_decay
    weight_decay
    '''
    from inspect import getsource
    getsource = getsource(opts.__class__)
    print getsource


if __name__ == "__main__":
    fire.Fire()
