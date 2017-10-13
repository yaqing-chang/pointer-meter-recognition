# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:49:27 2017

@author: caiwd

##from picture get dial

##num is a list ,every param is the number dials of every lines

##input : a opencv read array
"""
import cv2
import time
import os
import sys
import multiprocessing
from multiprocessing import Process,Queue,Pool
import numpy as np
import pymysql
from model import Universal_value


class Main(Universal_value):
    def __init__(self):
        Universal_value.__init__(self)
        self.itera = 0
        self.large_one = self.pic_size[0]*self.pic_size[1]
        self.minRadius = 150
        self.maxRadius=int(1.5*self.minRadius)
        self.quene_list()
        for camera_num in range(self.camera_nums):
            self.empty_piclist(camera_num)
        #self.sift_template()
        self.no_match_0 = False
        self.no_match_1 = False
        self.no_match_2 = False
        self.no_match_3 = False
        self.no_match_4 = False
        self.no_match_5 = False
        self.no_match_6 = False
        self.no_match_7 = False
        

    def quene_list(self):
        for camera_num in range(self.camera_nums):
            exec('self.q_readvideo_%s = multiprocessing.Queue()'%camera_num)
            exec('self.q_pic_%s = multiprocessing.Queue()'%camera_num)
            
        
    def empty_piclist(self,camera_num):
        for dial_num in range(self.dial_num_list[camera_num][0]+self.dial_num_list[camera_num][1]):
            exec('self.memory_pic_%s_%s = []'%(camera_num,dial_num))
        
    def get_coordinate(self,img):
        circles_xyr = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,2*self.minRadius,param1=50,param2=30,minRadius=self.minRadius,maxRadius=self.maxRadius)
        object_num = self.num_dial
        xyr = circles_xyr[0][:object_num]
        xyr = sorted(xyr,key=lambda xyr:xyr[1])
        xyr_0 = xyr[:self.pic_num[0]]
        xyr_1 = xyr[self.pic_num[0]:]
        xyr_0 = sorted(xyr[:4],key=lambda xyr_0:xyr_0[0])
        xyr_1 = sorted(xyr[4:],key=lambda xyr_1:xyr_1[0])
        sorted_xyr = xyr_0 + xyr_1
        return sorted_xyr

    def sift_match(self,img1_gray,camera_num,dial_num):
        img2 = cv2.imread(r'template\%s_%s.jpg'%(camera_num,dial_num))
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)    
        sift= cv2.xfeatures2d.SIFT_create()
        kp1,des1 = sift.detectAndCompute(img1_gray, None)
        kp2,des2 = sift.detectAndCompute(img2_gray, None)  
        # BFmatcher with default parms  
        bf = cv2.BFMatcher(cv2.NORM_L2)  
        matches = bf.knnMatch(des1, des2, k=2)
        ratio = 0.5
        mkp1,mkp2 = [],[] 
        for m in matches:  
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:  
                m = m[0]  
                mkp1.append( kp1[m.queryIdx] )  
        p1 = np.float32([kp.pt for kp in mkp1])  
        #print ('match point num:',len(p1))
        return p1
        

    def readvideo(self,camera_num):
        cap = cv2.VideoCapture('rtsp://admin:bhxz2017@%s:554/h264/ch1/main/av_stream'%self.ip_address[camera_num])
        #cap = cv2.VideoCapture('1.avi')
        success,frame_video=cap.read()
        while success:  
            exec('self.q_readvideo_%s.put(frame_video)'%camera_num)
            success,frame_video=cap.read()
            #print ("-------",self.q_readvideo_0.qsize())
            #cv2.imshow('ss',frame_video)
            #cv2.waitKey(1)

    def to_memory(self,xyr,img,camera_num,padding = 10):
        n = 0
        large_one = self.large_one
        if self.itera == self.batch_size:####每次识别的图片张数
            self.itera = 0
            for dial_num in range(self.dial_num_list[camera_num][0]+self.dial_num_list[camera_num][1]):
                exec("self.q_pic_%s.put(self.memory_pic_%s_%s)"%(camera_num,camera_num,dial_num))
            #print ('Camera %s to memory ok!'%camera_num)
            self.empty_piclist(camera_num)
            #print ("Camera %s:"%camera_num,eval('self.q_pic_%s.qsize()'%camera_num))
        for i in xyr:
            if eval('self.no_match_%s == False'%n):
                i = list(map(int,i))
                new_img = img[(i[1]-i[2])-padding:(i[1]+i[2])+padding,(i[0]-i[2])-padding:(i[0]+i[2])+padding]
                #cv2.imwrite('%s_%s.jpg'%(camera_num,n),new_img)
                if self.itera == 0 or self.itera == 50:
                    if len(self.sift_match(new_img,camera_num,n)) >= 0:
                        new_img = (cv2.resize(new_img,self.pic_size)/255.0).reshape(1,large_one)
                        exec("self.memory_pic_%s_%s.append(new_img[0])"%(camera_num,n))
                        exec('self.no_match_%s = False'%n)
                    else:
                        exec('self.no_match_%s = True'%n)
                else:
                    new_img = (cv2.resize(new_img,self.pic_size)/255.0).reshape(1,large_one)
                    exec("self.memory_pic_%s_%s.append(new_img[0])"%(camera_num,n))  
            else:
                pass
            n += 1
        self.itera += 1
            
    def video_image(self,camera_num):
        p_video = multiprocessing.Process(target = self.readvideo,args = (camera_num,))
        p_video.start()
        while eval('self.q_readvideo_%s.empty()'%camera_num):
            time.sleep(0.1)
        print ('Camera %s, Init ok!!!'%camera_num)
        frame = eval('self.q_readvideo_%s.get()'%camera_num)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #eval('self.xyr_%s = get_coordinate(frame)')%camera_num
        exec('self.xyr_%s = [[600,500,190],[1200,550,168],[1800,600,150],[2000,550,150],[600,900,190],[1100,940,168],[1700,880,150],[2000,900,150]]'%camera_num)
        while True:
            xyr = eval('self.xyr_%s'%camera_num)
            self.to_memory(xyr,frame,camera_num)
            frame = eval('self.q_readvideo_%s.get()'%camera_num)
            #print ('agsha',self.q_readvideo_0.qsize())
            try:####for test
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except:
                print ('error!')

    def save_result_db(self,data):
        conn = pymysql.connect(user='root', passwd='bhxz2017', db='bhxz')
        cursor = conn.cursor()
        '''
        sql = 'CREATE TABLE result (%s, %s, %s, %s)'
        cursor.executemany(sql, data) 
        conn.commit()
        '''
        
        sql = 'INSERT INTO result VALUES (%s, %s, %s, %s)'
        cursor.executemany(sql, data) 
        conn.commit()
        conn.close()

    def all_camera(self):
        for camera_num in range(self.camera_nums):
            multiprocessing.Process(target=self.video_image,args = (camera_num,)).start()
     

    def tensorflow_gpu(self):
        parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(parentdir,'model')
        sys.path.insert(0,model_dir)
        from model import model2
        model = model2()
        while True:
            all_camera_datas = []
            for camera_num in range(self.camera_nums):
                print ("Camera %s:"%camera_num,eval('self.q_pic_%s.qsize()'%camera_num))
                for dial_num in range(self.dial_num_list[camera_num][0]+self.dial_num_list[camera_num][1]):
                    value = eval('self.q_pic_%s.get()'%(camera_num))
                    if len(value) == 0:
                        pass
                    else:
                        now_time = time.strftime('%H:%M:%S',time.localtime())
                        result = model.use_model(value,dial_num)
                        result_datas = ''
                        for result_data in result:
                            result_datas = result_datas + ' ' + str(result_data)
                        one_dial_data = (camera_num,dial_num,now_time,result_datas)
                        all_camera_datas.append(one_dial_data)
            self.save_result_db(all_camera_datas)
            

time.clock()        
if __name__ == '__main__':
    start = Main()
    p_pic = multiprocessing.Process(target=start.tensorflow_gpu)
    p_pic.start()
    start.all_camera()
    print (time.clock())



    




    
