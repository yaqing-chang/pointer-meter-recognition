
##move different dial together,them larger it
import os
import shutil
import re
import numpy as np
import cv2
import random
from pro_data import transform_image,noise,one_pics_change

def get_pic_name(dial_num,class_dial):
    file_name  = os.listdir(r'template\{0}\{1}'.format(dial_num,class_dial))
    file_name = [x for x in file_name if re.match(r'\d{1,3}.jpg',x)]
    return file_name

def copy_file(dial_num,class_dial,pic_name):
    if os.path.exists(r'template\AllInOne\{}'.format(class_dial)):
        pass
    else:
        os.makedirs(r'template\AllInOne\{}'.format(class_dial))
    pic_num = len(os.listdir(r'template\AllInOne\{}'.format(class_dial)))
    shutil.copy(r'template\{0}\{1}\{2}'.format(dial_num,class_dial,pic_name),'template\AllInOne\{0}\{1}.jpg'.format(class_dial,pic_num))

def onetype_propic(template_num,name):
    class_num = len(os.listdir('template\%s'%name))
    for class_index in range(class_num):
        nums = len(os.listdir('template\{}\{}'.format(name, class_index)))
        for j in range(template_num-nums):
            randomnum = np.random.randint(0,nums)
            image = cv2.imread(r'template\%s\%d\%d.jpg'%(name, class_index, randomnum))
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
            cv2.imwrite(r'template\%s\%d\%d.jpg'%(name, class_index, nums+j),img)

    
all_file = os.listdir('template')
for dial_num in [i for i in all_file if re.match(r'\d',i)]:
    onetype_propic(20,dial_num)
    print ('Dial %s generate well!'%dial_num)
    class_dial = os.listdir(r'template\{}'.format(dial_num))
    for class_dial in [j for j in class_dial if re.match(r'\d',j)]:
        if os.path.exists(r'template\{0}\{1}'.format(dial_num,class_dial)):
            file_name = get_pic_name(dial_num,class_dial)
            for pic_name in file_name:
                copy_file(dial_num,class_dial,pic_name)
        else:
            pass
'''
if __name__ == '__main__':        
    onetype_propic(100)

'''

    
