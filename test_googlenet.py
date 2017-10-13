# coding: utf-8
import numpy as np
import time

from PIL import Image

from lib import data_util
from lib.config import params_setup
from lib.googlenet import GoogLeNet

# model
# scope_name, label_size = '17flowers', 17
# scope_name, label_size = '17portraits', 9
args = params_setup()
gnet = GoogLeNet(args=args)


#---------------------------
#   Server
#---------------------------
def guess():
    img = Image.open(r'.\images\17flowers\jpg\15\0001.jpg')
    img = img.resize((224,224), Image.ANTIALIAS).convert('L')
    #img = img.resize((164,164), Image.ANTIALIAS)
    #img.show()
    img = np.asarray(img, dtype="float32")
    img = img.reshape((224,224,1))
    img /= 255.0
    a = time.clock()
    imgs = []
    for i in range(100):
        imgs.append(img)
    b = time.clock()
    probs = gnet.predict_label(imgs)
    #print (time.clock()-b)
    print (probs)

if __name__ == '__main__':
    aa = time.clock()
    for i in range(20):
        guess()
    print (time.clock() - aa)

