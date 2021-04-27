#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 
# * Create time   : 2018-08-13 09:41
# * Last modified : 2018-08-13 10:02
# * Filename      : config.py
# * Description   : this part is config  
# **********************************************************
import argparse

def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("-r",'--root',help='train root',default = './data/train')
    arg.add_argument("-t",'--test',help = 'test root',default = './data/test')
    arg.add_argument("-b","--batch",type = int,default = 128)
    arg.add_argument("-e","--epoch",type = int,default = 60)
    arg.add_argument("-l","--lr",type = float,default = 1e-3)
    arg.add_argument("-c","--config",type = bool,default = False)
    para = arg.parse_args()
    return para


class DefaultConfig(object):
    env = 'default' #visdom
    iteration = 100000
    lr = 1e-10
    momentum = 0.99
    weight_decay = 0.0005
    interval_validate = 4000
    load_model_path = './check_point/FCN_0905_04:42:42.pt'
    model ="FCN8s_At_Once"
    pretrain = True
    batch_size = 16
    use_gpu = True
    def __init__(self,args = None):
        if args == None:
            return 
        self.parse(args)
    def parse(self,args):
        '''
        re config the para 
        '''
        if args.config:
            for k,v in args.__get_kwargs():
                if not hasattr(self,k):
                    print("warning: opt has nor attribute {}".format(k))
                setattr(self,k,v)

        else:
            pass
        # print config
        print('user config:')
        for k,v in self.__class__.__dict__.iteritems():
            if not k.startswith("__") and k !='parse':
                print("{} : {}".format(k,getattr(self,k)))

    def parse_kwargs(self,**kwargs):
        for k,v in kwargs.iteritems():
            if not hasattr(self,k):
                print("warning opt has nor attribute{}".format(k))
            setattr(self,k,v)

        print('here user config:')
        for k,v in self.__class__.__dict__.iteritems():
            if not k.startswith("__") and k !='parse' and k!='parse_kwargs':
                print("{} : {}".format(k,getattr(self,k)))

def config_get():
    arg = get_args()
    config =DefaultConfig(arg)
    return config,arg
