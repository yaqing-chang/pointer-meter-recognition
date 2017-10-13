# -*- coding: utf8 -*-
import cv2,time,random
import numpy as np
import os,shutil
from PIL import Image

def transform_image(img,ang_range,shear_range,trans_range):    
    #This function transforms images to generate new images.
    #The function takes in following arguments,
    #1- Image
    #2- ang_range: Range of angles for rotation
    #3- shear_range: Range of values to apply affine transform to
    #4- trans_range: Range of values to apply translations over.     
    #A Random uniform distribution is used to generate different parameters for transformation    
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)#(旋转中心，旋转角度，放缩倍数)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    return img

def noise(count,img):
    count = np.random.randint(count)
    for k in range(0,count):
        #get the random point
        xi = int(np.random.uniform(0,img.shape[1]))
        xj = int(np.random.uniform(0,img.shape[0]))
        #add noise
        if img.ndim == 2:
            img[xj,xi] = 255
        elif img.ndim == 3:
            img[xj,xi,0] = np.random.randint(0,100)
            img[xj,xi,1] = np.random.randint(0,100)
            img[xj,xi,2] = np.random.randint(0,100)


def one_pics_change(dial_type,pic_num,dir):
    parent_dir = os.path.split(os.path.realpath(__file__))[0]
    child_dir = r'%s\%s'%(dir,dial_type)
    dirs = os.path.join(parent_dir,child_dir)
    class_num = len(os.listdir(dirs))
    for i in range(class_num):
        nums = len(os.listdir(os.path.join(parent_dir,r'%s\%s\%s'%(dir,dial_type,i))))
        num = nums
        for j in range(pic_num-nums):
            randomnum = np.random.randint(0,nums)
            image = cv2.imread(os.path.join(parent_dir,r'%s\%s\%d\%d.jpg'%(dir,dial_type,i,randomnum)))
            width = image.shape[0]
            height = image.shape[1]
            lightnum = 30
            lightnum = random.randint(-lightnum,(lightnum))
            img = image+lightnum+random.randint(-10,10)
            img = transform_image(img,30,0,20)
            noise(1000,img)
            x_offset = np.random.randint(0.15*width)
            y_offset = np.random.randint(0.15*height)
            width = int(0.85*width)
            height = int(0.85*height)
            img = img[x_offset:(x_offset+width),y_offset:(y_offset+height)]
            if not np.random.randint(0,3):
                if  np.random.randint(0,2):
                    a = 3
                else :
                    a = 5
                b = np.random.randint(0,2)
                img = cv2.GaussianBlur(img, (a, a), b);
            cv2.imwrite(os.path.join(parent_dir,r'%s\%s\%d\%d.jpg'%(dir,dial_type,i,num)),img)
            num +=1    


def onetype_propic(dial_type,pic_num_train,pic_num_test):
    parent_dir = os.path.split(os.path.realpath(__file__))[0]
    child_dir = r'template\%s\%s'%(dial_type,0)
    dirs = os.path.join(parent_dir,child_dir)
    template_num = len(os.listdir(dirs))
    template_test = int(0.2*template_num)
    template_train = template_num-template_test
    class_num = len(os.listdir(os.path.join(parent_dir,r'template\%s'%dial_type)))
    for i in range(class_num):
        for j in range(template_train):
            if os.path.exists(os.path.join(parent_dir,r'train\%s\%s'%(dial_type,i))):
                pass
            else:
                os.makedirs(os.path.join(parent_dir,r'train\%s\%s'%(dial_type,i)))
            shutil.copyfile(os.path.join(parent_dir,r'template\%s\%s\%s.jpg'%(dial_type,i,j)), os.path.join(parent_dir,r'train\%s\%s\%s.jpg'%(dial_type,i,j)))
        for j in range(template_test):
            if os.path.exists(os.path.join(parent_dir,r'test\%s\%s'%(dial_type,i))):
                pass
            else:
                os.makedirs(os.path.join(parent_dir,r'test\%s\%s'%(dial_type,i)))
            shutil.copyfile(os.path.join(parent_dir,r'template\%s\%s\%s.jpg'%(dial_type,i,j+template_train)), os.path.join(parent_dir,r'test\%s\%s\%s.jpg'%(dial_type,i,j)))
    print ('generate train pic--%s  ...'%dial_type)
    one_pics_change(dial_type,pic_num_train,'train')
    print ('generate test pic--%s ...'%dial_type)
    one_pics_change(dial_type,pic_num_test,'test')
if __name__ == '__main__':
    onetype_propic(dial_type=0,pic_num_train=100,pic_num_test=5)
    



