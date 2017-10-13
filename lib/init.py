# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:17:18 2017

@author: caiwd
"""
from . import  config

config = config.params_setup()
class Universal_value():
    def __init__(self):
        self.pic_size = (config.img_size,config.img_size)
        self.class_num = config.label_size
        self.batch_size = config.batch_size
        self.ip_address = config.device_ip
        self.dial_num_list = config.dial_num_list
        self.camera_nums = len(self.ip_address)
        

