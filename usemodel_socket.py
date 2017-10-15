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


from lib.init import Universal_value


class Main(Universal_value):
    def __init__(self):
        Universal_value.__init__(self)
        self.itera = 0
        self.minRadius = 220
        self.maxRadius=int(1.2*self.minRadius)
        self.quene_list()
        for camera_num in range(self.camera_nums):
            self.empty_piclist(camera_num)
            for dial_num in range(self.dial_num_list[camera_num][0]+self.dial_num_list[camera_num][1]):
                exec('self.match_%s_%s = True'%(camera_num,dial_num))
        

    def empty_piclist(self,camera_num):
        for dial_num in range(self.dial_num_list[camera_num][0]+self.dial_num_list[camera_num][1]):
            exec('self.memory_pic_%s_%s = []'%(camera_num,dial_num))        

    def quene_list(self):
        self.q_result_socket_db = multiprocessing.Queue(20)
        for camera_num in range(self.camera_nums):
            exec('self.q_readvideo_%s = multiprocessing.Queue()'%camera_num)
            exec('self.q_pic_%s = multiprocessing.Queue()'%camera_num)
    

    def sift_match(self,img1_gray,camera_num,dial_num):
        img2 = cv2.imread(r'template\%s_%s.jpg'%(camera_num,dial_num))
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)    
        sift= cv2.xfeatures2d.SIFT_create()
        kp1,des1 = sift.detectAndCompute(img1_gray, None)
        kp2,des2 = sift.detectAndCompute(img2_gray, None)  
        # BFmatcher with default parms  
        bf = cv2.BFMatcher(cv2.NORM_L2)  
        matches = bf.knnMatch(des1, des2, k=2)
        ratio = 0.45
        mkp1,mkp2 = [],[] 
        for m in matches:  
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:  
                m = m[0]  
                mkp1.append( kp1[m.queryIdx] )  
        p1 = np.float32([kp.pt for kp in mkp1])  
        #print ('match point num:',len(p1))
        return p1

    
    def init(self):
        for camera_num in range(self.camera_nums):
            #cap = cv2.VideoCapture('rtsp://admin:bhxz2017@%s:554/h264/ch1/main/av_stream'%self.ip_address[camera_num])
            cap = cv2.VideoCapture('%s.avi'%camera_num)
            success, frame = cap.read()
            if success:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('1.jpg',frame)
                if os.path.exists('coordinate.ini'):
                    with open('coordinate.ini','r') as f:
                        lines = len(f.readlines())
                        if (lines == self.camera_nums):
                            pass
                        elif (lines >= self.camera_nums):
                            f.close()
                            os.remove('coordinate.ini')
                            self.get_coordinate(camera_num,frame_gray)
                        else :
                            self.get_coordinate(camera_num,frame_gray)
                else :
                    self.get_coordinate(camera_num,frame_gray)
                xyr = self.read_coordinate(camera_num)
                for i in xyr:
                    cv2.rectangle(frame,((i[0]-i[2]),(i[1]-i[2])),((i[0]+i[2]),(i[1]+i[2])),(0,255,155),5)
                new_img = cv2.resize(frame, None, fx = 0.3, fy = 0.3)
                cv2.imshow('Detection',new_img)
                cv2.waitKey(0)
            else :
                print ("[NO.%s] @%s Can't Connect To Camera!"%(camera_num,self.ip_address[camera_num]))
                time.sleep(30)
                os._exit(0)


    def read_coordinate(self,camera_num):
        import re
        xyr = []
        with open('coordinate.ini','r') as f:
            for lines in range(camera_num+1):
                a = f.readline().strip('\n')
            coordinate = list(x for x in re.split(r'[\[\]]',a) if x)
            for i in coordinate:
                xyr.append(list(map(int,i.split(','))))
        return xyr
    

    def get_coordinate(self,camera_num,img):
        from functools import reduce
        circles_xyr = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,2*self.minRadius,param1=50,param2=30,minRadius=self.minRadius,maxRadius=self.maxRadius)
        dial_num = self.dial_num_list[camera_num][0]+self.dial_num_list[camera_num][1]
        xyr = circles_xyr[0][:dial_num]
        xyr = sorted(xyr,key=lambda xyr:xyr[1])
        xyr_0 = xyr[:self.dial_num_list[camera_num][0]]
        xyr_1 = xyr[self.dial_num_list[camera_num][0]:]
        xyr_0 = sorted(xyr_0,key=lambda xyr_0:xyr_0[0])
        xyr_1 = sorted(xyr_1,key=lambda xyr_1:xyr_1[0])
        sorted_xyr = xyr_0 + xyr_1
        means_radius = int(reduce((lambda x,y:x+y),sorted_xyr)[2]/(self.dial_num_list[camera_num][0]+self.dial_num_list[camera_num][1]))
        for i in sorted_xyr:
            i = list(map(int,i))
            i[2] = means_radius
            with open('coordinate.ini','a') as f:
                f.write(str(i))
        with open('coordinate.ini','a') as f:
            f.write('\n')

        
    def readvideo(self,camera_num):
        #cap = cv2.VideoCapture('rtsp://admin:bhxz2017@%s:554/h264/ch1/main/av_stream'%self.ip_address[camera_num])
        cap = cv2.VideoCapture('%s.avi'%camera_num)
        success,frame_video=cap.read()
        while success:  
            exec('self.q_readvideo_%s.put(frame_video)'%camera_num)
            success,frame_video=cap.read()
            #print ("VideoToPicture: ",self.q_readvideo_0.qsize())
            #cv2.imshow('VideoToPicture',frame_video)
            #cv2.waitKey(1)

    def dail_pic_to_memory(self,xyr,img,camera_num,padding = 10):
        n = 0
        for i in xyr:
            i = list(map(int,i))
            new_img = img[(i[1]-i[2])-padding:(i[1]+i[2])+padding,(i[0]-i[2])-padding:(i[0]+i[2])+padding]
            #cv2.imwrite('%s_%s.jpg'%(camera_num,n),new_img)
            if self.itera == 0 or self.itera == 50:
                match_point = len(self.sift_match(new_img,camera_num,n))
                if match_point >= 0:
                    #print ('Match Point (%s-%s): %s'%(camera_num,n,match_point))
                    exec('self.match_%s_%s = True'%(camera_num,n))
                else:
                    #print("      Can't Matct (%s-%s): %s"%(camera_num,n,match_point))
                    exec('self.match_%s_%s = False'%(camera_num,n))
            if eval('self.match_%s_%s == True'%(camera_num,n)):
                new_img = (cv2.resize(new_img,self.pic_size)/255.0).reshape((self.pic_size[0],self.pic_size[0],1))
                exec("self.memory_pic_%s_%s.append(new_img)"%(camera_num,n))
            n += 1

        if self.itera == self.batch_size:
            self.itera = 0
            for dial_num in range(self.dial_num_list[camera_num][0]+self.dial_num_list[camera_num][1]):
                exec("self.q_pic_%s.put(self.memory_pic_%s_%s)"%(camera_num,camera_num,dial_num))
            self.empty_piclist(camera_num)
            print ("Camera %s:"%camera_num,eval('self.q_pic_%s.qsize()'%camera_num))
        self.itera += 1
            
    def video_image(self,camera_num):
        p_video = multiprocessing.Process(target = self.readvideo,args = (camera_num,))
        p_video.start()
        while eval('self.q_readvideo_%s.empty()'%camera_num):
            time.sleep(0.1)
        print ('Camera %s, Init ok!!!'%camera_num)
        frame = eval('self.q_readvideo_%s.get()'%camera_num)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        exec('self.xyr_%s = self.read_coordinate(camera_num)'%camera_num)
        #exec('self.xyr_%s = [[600,500,190],[1200,550,168],[1600,600,150],[1800,550,150]]'%camera_num)
        while True:
            xyr = eval('self.xyr_%s'%camera_num)
            self.dail_pic_to_memory(xyr,frame,camera_num)
            frame = eval('self.q_readvideo_%s.get()'%camera_num)
            #print ('agsha',self.q_readvideo_0.qsize())
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except:
                print ('error!')

    def all_camera(self):
        #if self.Initialization:
        if True:
            self.init()
        for camera_num in range(self.camera_nums):
            multiprocessing.Process(target=self.video_image,args = (camera_num,)).start()

            

    def save_result_db(self,data):
        conn = pymysql.connect(user='root', passwd='941120', db='bhxz')
        cursor = conn.cursor()
        '''
        sql = 'CREATE TABLE result (%s, %s, %s)'
        cursor.executemany(sql, data) 
        conn.commit()
        '''
        times = data['time']
        data.pop('time')
        sql = 'INSERT INTO test VALUES (%s, %s, %s)'
        datas = [(times,y,data[y]) for y in data]
        cursor.executemany(sql, datas) 
        conn.commit()
        conn.close()


    def send_socket(self):
        import socket
        import json
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        localIP = socket.gethostbyname(socket.gethostname())
        server.bind(('%s'%localIP, 5000))
        server.listen(5)
        while True:
            print ('Waiting for Connection......')
            con, address = server.accept()
            print ('Connected by %s!'%address[0])
            while True:
                all_camera_datas = self.q_result_socket_db.get()
                send_data = json.dumps(all_camera_datas)
                try:
                    con.send(send_data.encode())
                    time.sleep(1)
                except:
                    break
     
                     
    def error_data(self,result_datas):
        n = 0
        for i in result_datas:
            try:
                if abs(result_datas[n+1]-result_datas[n]) > 2:
                    #result_datas = np.delete(result_datas,n+1)
                    result_datas[n+1] = int(result_datas[n])
                n += 1
            except:
                pass
        return result_datas

    def tensorflow_gpu(self):
        #from test_googlenet import gnet
        from lib.Alexnet import model2
        model = model2()
        multiprocessing.Process(target= (self.send_socket), args = ()).start()
        while True:
            #print ("Camera %s:"%camera_num,eval('self.q_pic_%s.qsize()'%camera_num))
            all_camera_datas = {}
            now_time = time.time()
            all_camera_datas['time'] = round(now_time,3)
            for camera_num in range(self.camera_nums):
                for dial_num in range(self.dial_num_list[camera_num][0]+self.dial_num_list[camera_num][1]):
                    value = eval('self.q_pic_%s.get(True)'%(camera_num))
                    if len(value) == 0:
                        print ('No data for model, %s-%s！'%(camera_num,dial_num))
                    else:
                        #result_datas = gnet.predict_label(value)
                        result_datas = model.use_model(value,dial_num)
                        print(result_datas)
                        result_datas = self.error_data(result_datas)
                        name = '%s-%s'%(camera_num,dial_num)
                        all_camera_datas[name] = str(result_datas).strip('[]')
            self.save_result_db(all_camera_datas)
            if self.q_result_socket_db.full():
                self.q_result_socket_db.get()
            self.q_result_socket_db.put(all_camera_datas)
          
       
if __name__ == '__main__':
    start = Main()
    p_pic = multiprocessing.Process(target=start.tensorflow_gpu)
    p_pic.start()
    start.all_camera()









    
