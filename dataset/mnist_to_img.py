#conding:utf8
import numpy as np
import struct,gzip,cv2
import matplotlib.pyplot as plt
 
filename = 'test_label_ubyte.gz'
binfile = gzip.open(filename , 'rb')
buf = binfile.read()
 
index = 0
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
print (magic,numImages,numRows , numColumns)
index += struct.calcsize('>II')
for i in range(1000) :
    num = numRows * numColumns
    num = 1
    im = struct.unpack_from('>%sB'%num ,buf, index)
    index += struct.calcsize('>%sB'%num)
    print (im)
    '''
    im = np.array(im)
    im = im.reshape(numRows,numColumns)
    im = im/255.0

    cv2.imshow('z',im)
    cv2.waitKey(0)
    '''
    '''
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im , cmap='gray')
    plt.show()
    '''
