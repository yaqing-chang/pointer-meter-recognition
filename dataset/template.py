#coding:utf8
# for  to pro different pic
import time
import os
from PIL import Image


class pro_template():
    def __init__(self,test,dial_type,class_num,angle_list):
        self.dial_type = dial_type
        self.class_num = class_num
        self.test = test
        self.angle_list = angle_list
        if test:
            self.test_sig = 10
        else:
            self.test_sig = 1

    def transPNG(self,srcImageName):
        img = Image.open(srcImageName)
        img = img.convert("RGBA")
        datas = img.getdata()
        newData = list()
        for item in datas:
            if item[0] >180 and item[1] > 180 and item[2] > 180:
                newData.append(( item[0], item[1], item[2], 0))
            else:
                newData.append(item)    
        img.putdata(newData)
        return img

    def angle(self,i):
        num_0 = int(i/10)
        num_1 = i%10
        angle = 0
        for j in range(num_0):
            angle += self.angle_list[j]*10
        angle += num_1*self.angle_list[num_0]
        return angle
    
    def main(self,offset=(2,1),bppic = 'bp.jpg',zzpic = 'zz.jpg'):
        class_num = self.class_num
        test_sig = self.test_sig
        for i in range(0,class_num,test_sig):
            parent_dir = os.path.split(os.path.realpath(__file__))[0]
            child_dir = r'template\%s\%s'%(self.dial_type,i)
            dirs = os.path.join(parent_dir,child_dir)
            try :
                os.makedirs(dirs)
                num = 0
            except:
                num = len(os.listdir(dirs))
            bppic = os.path.join(parent_dir,bppic)
            im = Image.open(bppic)
            im = im.convert('RGBA')
            zzpic = os.path.join(parent_dir,zzpic)
            img = self.transPNG(zzpic)
            img = img.rotate(12-self.angle(i))
            r,g,b,a = img.split()
            im.paste(img,offset,mask=a)
            if test_sig == 10:
                test_dir = r'template\test\%s.jpg'%i
                dirs = os.path.join(parent_dir,test_dir)
                im.save(dirs)
            im = im.resize((400,400))
            if test_sig == 1:
                save_dir = r'template\%s\%s\%s.jpg'%(self.dial_type,i,num)
                dirs = os.path.join(parent_dir,save_dir)
                im.save(dirs)
            im.close()
        
if __name__=='__main__':
    a = pro_template(test=False,
                 dial_type=0,
                 class_num=101,
                 angle_list = [2.28,2.21,2.22,2.22,2.21,2.21,2.21,2.14,2.14,2.17,2.18,2.33,2.25])
    a.main()

