# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:17:18 2017

@author: caiwd
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import xlwt
import xlrd
from xlutils.copy import copy  
import time
import os
from . import init
from . import readdata

Universal_value = init.Universal_value

match_model_map = {'0-0':1,'0-1':1,'0-2':1,'0-3':1,'0-4':1,'0-5':1,'0-6':1,'0-7':1}
    


class model(Universal_value):
    def __init__(self,
                 input_img_size=(192,192),
                 output_num=101,
                 nkerns=[48,128,196,256,338],
                 seq = 0,
                 num_dial = 1):
        Universal_value.__init__(self)
        input_img_size = self.pic_size
        output_num = self.class_num
        self.xs = tf.placeholder(tf.float32, [None, input_img_size[0], input_img_size[1], 1])
        self.ys = tf.placeholder(tf.float32, [None, output_num])
        self.keep_prob = tf.placeholder(tf.float32)
            
        x_image = self.xs
        nkerns = nkerns

        W_conv1 = self.weight_variable([5,5, 1,nkerns[0]]) 
        b_conv1 = self.bias_variable([nkerns[0]])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1) 
        h_pool1 = self.max_pool_2x2(h_conv1)  

        W_conv2 = self.weight_variable([3,3, nkerns[0], nkerns[1]]) 
        b_conv2 = self.bias_variable([nkerns[1]])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) 
        h_pool2 = self.max_pool_2x2(h_conv2)        

        W_conv3 = self.weight_variable([3,3, nkerns[1], nkerns[2]]) 
        b_conv3 = self.bias_variable([nkerns[2]])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)        

        W_conv4 = self.weight_variable([3,3, nkerns[2], nkerns[3]]) 
        b_conv4 = self.bias_variable([nkerns[3]])
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3, W_conv4) + b_conv4) 
        h_pool4 = self.max_pool_2x2(h_conv4)        

        W_conv5 = self.weight_variable([3,3, nkerns[3], nkerns[4]]) 
        b_conv5 = self.bias_variable([nkerns[4]])
        h_conv5 = tf.nn.relu(self.conv2d(h_pool4, W_conv5) + b_conv5) 
        h_pool5 = self.max_pool_2x2(h_conv5)

        W_fc1 = self.weight_variable([4*4*nkerns[4],800])
        b_fc1 = self.bias_variable([800])
        h_pool2_flat = tf.reshape(h_pool5, [-1, 4*4*nkerns[4]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob) 

        W_fc11 = self.weight_variable([800,400])
        b_fc11 = self.bias_variable([400])
        h_pool21_flat = tf.reshape(h_fc1_drop, [-1, 800])
        h_fc11 = tf.nn.relu(tf.matmul(h_pool21_flat, W_fc11) + b_fc11)
        h_fc11_drop = tf.nn.dropout(h_fc11, self.keep_prob) 

        W_fc2 = self.weight_variable([400, output_num])
        b_fc2 = self.bias_variable([output_num])
        Ylogits = tf.matmul(h_fc11_drop, W_fc2) + b_fc2
        self.prediction = tf.nn.softmax(Ylogits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=self.ys)
        self.train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
        self.y_pre = tf.argmax(self.prediction,1)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.seq = seq
        self.data = time.strftime("%m-%d", time.localtime())
        
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_2x2(self,x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    def compute_accuracy(self,batch_size):
        results = 0.0
        for i in range(self.n_test_batches):
            v_xs, v_ys = self.dataset.test.next_batch(batch_size)
            y_pre = self.sess.run(self.prediction, feed_dict={self.xs: v_xs, self.keep_prob: 1})
            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result = self.sess.run(accuracy, feed_dict={self.xs: v_xs, self.ys: v_ys, self.keep_prob: 1})
            results +=result
        return results/float((i+1))

    def use_model(self,img,camera_num, dial_type = 0):
        #self.data = time.strftime("%m-%d", time.localtime())
        dial_type = match_model_map['{}-{}'.format(camera_num, dial_type)]
        try:
            self.saver.restore(self.sess, r'.\param\Alexnet\%s\data.ckpt'%dial_type)
        except:
            print ('Restory Model Error!')
        from datetime import datetime
        result = self.sess.run(self.y_pre,feed_dict={self.xs: img, self.keep_prob: 1})
        #print (time.strftime('%H:%M:%S',time.localtime()),result)
        

    def train_model(self,dial_type=0,epoch_num=50,batch_size=40):
        dirs = os.getcwd()
        size = self.pic_size[0]
        dataset_dir = r'dataset\datasetwell\%s\size-%s'%(dial_type,size)
        dataset_dir = os.path.join(dirs,dataset_dir)
        alldata = readdata.read_data_sets(dataset_dir, one_hot=True)
        self.dataset = alldata[0]
        self.n_train_batches = int(alldata[1]/batch_size)
        self.n_test_batches = int(alldata[3]/batch_size)
        bestmode = 0
        epoch = 0
        iters = 0
        for items in range(self.n_train_batches*epoch_num):
            batch_xs, batch_ys = self.dataset.train.next_batch(batch_size)
            self.sess.run(self.train_step, feed_dict={self.xs: batch_xs, self.ys: batch_ys, self.keep_prob: 0.8})
            if items % batch_size == 0:
                accuracy = self.compute_accuracy(batch_size)
                if accuracy > bestmode:
                    bestmode = accuracy
                    save_path = self.saver.save(self.sess,r'.\param\Alexnet\%s\data.ckpt'%dial_type)
                    print('     epoch %s'%epoch,' accuracy ',accuracy*100,' %')
                else:
                    print('epoch %s'%epoch,' accuracy ',accuracy*100,' %')
                epoch += 1
        print ('The Best Accuracy is %s'%bestmode)  


class model2(Universal_value):
    def __init__(self,
                 input_img_size=(164,164),
                 output_num=101,
                 nkerns=[48,128,196,256,338],
                 seq = 0,
                 num_dial = 1):
        Universal_value.__init__(self)
        input_img_size = self.pic_size
        output_num = self.class_num
        self.xs = tf.placeholder(tf.float32, [None, input_img_size[0], input_img_size[1], 1])
        self.ys = tf.placeholder(tf.float32, [None, output_num])
        self.keep_prob = tf.placeholder(tf.float32)
            
        x_image = self.xs
        nkerns = nkerns

        W_conv1 = self.weight_variable([5,5, 1,nkerns[0]]) 
        b_conv1 = self.bias_variable([nkerns[0]])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1) 
        h_pool1 = self.max_pool_2x2(h_conv1)  

        W_conv2 = self.weight_variable([5,5, nkerns[0], nkerns[1]]) 
        b_conv2 = self.bias_variable([nkerns[1]])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) 
        h_pool2 = self.max_pool_2x2(h_conv2)        

        W_conv3 = self.weight_variable([3,3, nkerns[1], nkerns[2]]) 
        b_conv3 = self.bias_variable([nkerns[2]])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)        

        W_conv4 = self.weight_variable([3,3, nkerns[2], nkerns[3]]) 
        b_conv4 = self.bias_variable([nkerns[3]])
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3, W_conv4) + b_conv4) 
        h_pool4 = self.max_pool_2x2(h_conv4)        

        W_conv5 = self.weight_variable([3,3, nkerns[3], nkerns[4]]) 
        b_conv5 = self.bias_variable([nkerns[4]])
        h_conv5 = tf.nn.relu(self.conv2d(h_pool4, W_conv5) + b_conv5) 
        h_pool5 = self.max_pool_2x2(h_conv5)

        W_fc1 = self.weight_variable([3*3*nkerns[4],800])
        b_fc1 = self.bias_variable([800])
        h_pool2_flat = tf.reshape(h_pool5, [-1, 3*3*nkerns[4]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob) 

        W_fc11 = self.weight_variable([800,400])
        b_fc11 = self.bias_variable([400])
        h_pool21_flat = tf.reshape(h_fc1_drop, [-1, 800])
        h_fc11 = tf.nn.relu(tf.matmul(h_pool21_flat, W_fc11) + b_fc11)
        h_fc11_drop = tf.nn.dropout(h_fc11, self.keep_prob) 

        W_fc2 = self.weight_variable([400, output_num])
        b_fc2 = self.bias_variable([output_num])
        Ylogits = tf.matmul(h_fc11_drop, W_fc2) + b_fc2
        self.prediction = tf.nn.softmax(Ylogits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=self.ys)
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        self.y_pre = tf.argmax(self.prediction,1)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.seq = seq
        self.data = time.strftime("%m-%d", time.localtime())
        
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_2x2(self,x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def compute_accuracy(self,batch_size):
        results = 0.0
        for i in range(self.n_test_batches):
            v_xs, v_ys = self.dataset.test.next_batch(batch_size)
            y_pre = self.sess.run(self.prediction, feed_dict={self.xs: v_xs, self.keep_prob: 1})
            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result = self.sess.run(accuracy, feed_dict={self.xs: v_xs, self.ys: v_ys, self.keep_prob: 1})
            results +=result
        return results/float((i+1))

    def use_model(self,img,camera_num, dial_type = 0):
        #self.data = time.strftime("%m-%d", time.localtime())
        dial_type = match_model_map['{}-{}'.format(camera_num, dial_type)]
        try:
            self.saver.restore(self.sess, r'.\param\Alexnet\%s\data.ckpt'%dial_type)
        except:
            print ('Restory Model Error!')
        from datetime import datetime
        result = self.sess.run(self.y_pre,feed_dict={self.xs: img, self.keep_prob: 1})
        return result
        

    def train_model(self,dial_type=0,epoch_num=50,batch_size=10):
        dirs = os.getcwd()
        size = self.pic_size[0]
        dataset_dir = r'dataset\datasetwell\%s\size-%s'%(dial_type,size)
        dataset_dir = os.path.join(dirs,dataset_dir)
        alldata = readdata.read_data_sets(dataset_dir, one_hot=True)
        self.dataset = alldata[0]
        self.n_train_batches = int(alldata[1]/batch_size)
        self.n_test_batches = int(alldata[3]/batch_size)
        bestmode = 0
        epoch = 0
        iters = 0
        print (self.n_train_batches)
        for items in range(self.n_train_batches*epoch_num):
            batch_xs, batch_ys = self.dataset.train.next_batch(batch_size)
            self.sess.run(self.train_step, feed_dict={self.xs: batch_xs, self.ys: batch_ys, self.keep_prob: 0.8})
            if items % self.n_train_batches == 0:
                accuracy = self.compute_accuracy(batch_size)
                if accuracy > bestmode:
                    bestmode = accuracy
                    save_path = self.saver.save(self.sess,r'.\param\Alexnet\%s\data.ckpt'%dial_type)
                    print('     epoch %s'%epoch,' accuracy ',accuracy*100,' %')
                else:
                    print('epoch %s'%epoch,' accuracy ',accuracy*100,' %')
                epoch += 1
        print ('the beat accuracy is %s'%bestmode)  

        
if __name__=='__main__':
    a = model2()
    a.train_model()
    










