
##move different dial together
import os
import shutil
import re

def get_pic_name(dial_num,class_dial):
    file_name  = os.listdir(r'{0}\{1}'.format(dial_num,class_dial))
    file_name = [x for x in file_name if re.match(r'\d{1,3}.jpg',x)]
    return file_name

def copy_file(dial_num,class_dial,pic_name):
    if os.path.exists(r'AllInOne\{}'.format(class_dial)):
        pass
    else:
        os.makedirs(r'AllInOne\{}'.format(class_dial))
    pic_num = len(os.listdir(r'AllInOne\{}'.format(class_dial)))
    shutil.copy(r'{0}\{1}\{2}'.format(dial_num,class_dial,pic_name),'AllInOne\{0}\{1}.jpg'.format(class_dial,pic_num))
    
all_file = os.listdir()
for dial_num in [i for i in all_file if re.match(r'\d',i)]:
    class_dial = os.listdir(r'{}'.format(dial_num))
    for class_dial in [j for j in class_dial if re.match(r'\d',j)]:
        if os.path.exists(r'{0}\{1}'.format(dial_num,class_dial)):
            file_name = get_pic_name(dial_num,class_dial)
            for pic_name in file_name:
                copy_file(dial_num,class_dial,pic_name)
        else:
            pass
