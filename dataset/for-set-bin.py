#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:14:27 2017

@author: caiwd
"""
import struct,gzip
import numpy as np
import os,pro_data
from PIL import Image
from ctypes import create_string_buffer

class bin_set():
    def __init__(self,train_num,test_num,dial_type,size):
        self.train_num = train_num#训练集每一种图片的张数
        self.test_num = test_num#测试集每一种图片的张数
        self.dial_type = dial_type#表盘的种类
        self.rows = size[0]
        self.columns = size[1]
        self.parent_dir = os.path.split(os.path.realpath(__file__))[0]    
        
    #将图片数据转为二进制文件
    def image(self,name,type,list0):
        image_num = len(list0)
        rows = self.rows
        columns = self.columns
        print ('%s'%name ,' set %s'%image_num,'pics')
        with gzip.open((os.path.join(self.parent_dir,r'datasetwell\%s\size-%s\%s_images_ubyte'%(type,rows,name)+'.gz')),'wb') as zipfile:
            buf = create_string_buffer(rows*columns*image_num+4*4)
            struct.pack_into('>4I',buf,0,2051,(image_num),rows,columns)
            m = 0
            for i in list0:
                im = Image.open((os.path.join(self.parent_dir,r'%s\%s\%s\%s.jpg'%(name,type,i[0],i[1])))).convert('L').resize((rows,columns))
                im = np.array(im)
                im = im.reshape(1,-1)
                #length = len(im)
                for j in range(rows*columns):
                    struct.pack_into('>B',buf,rows*columns*m+4*4+j,im[0][j])
                m += 1
            zipfile.write(buf)

    #将标签文件转换为二进制文件
    def label(self,name,type,list0):
        rows = self.rows
        conlumns = self.columns
        with gzip.open(os.path.join(self.parent_dir,r'datasetwell\%s\size-%s\%s_label_ubyte.gz'%(type,rows,name)),'w') as zipfile:
            label_num = len(list0)
            buf = create_string_buffer(label_num*1+2*4)
            n = 0
            for i in list0:
                try:
                    struct.pack_into('>B',buf,2*4+n,int(i[0]))
                except:
                    pass
                n += 1
            struct.pack_into('>2I',buf,0,2049,(label_num))
            zipfile.write(buf)

    def main(self):
        train = 'train'
        test = 'test'
        for type in range(self.dial_type):
            pro_data.onetype_propic(type,self.train_num,self.test_num)#生成图片
            try :
                os.makedirs(os.path.join(self.parent_dir,r'datasetwell\%s\size-%s'%(type,self.rows)))
            except:
                raise

            #生成保存图片顺序随机列表
            class_num = len(os.listdir(os.path.join(self.parent_dir,r'train\%s'%type)))            
            shufflelist_train = []
            shufflelist_test = []
            for j in range(class_num):
                for i in range(self.train_num):
                    temp_list = [[j,i]]
                    shufflelist_train += temp_list
            np.random.shuffle(shufflelist_train)
                          
            for j in range(class_num):
                for i in range(self.test_num):
                    temp_list = [[j,i]]
                    shufflelist_test += temp_list
            np.random.shuffle(shufflelist_test)

            #保存数据集
            self.image(train,type,shufflelist_train)
            self.image(test,type,shufflelist_test)
            self.label(train,type,shufflelist_train)   
            self.label(test,type,shufflelist_test)
            
        print ('successful')
if __name__ == '__main__':
    ##train_num为训练用的每种图片的张数
    ##test_num为测试用的每种图片的张数
    ##dial_type为表盘的种类数，应与template中的表盘种类的一致
    ##size为训练机中图片的大小
    a = bin_set(train_num=10,test_num=2,dial_type=4,size=(164,164))
    a.main()









