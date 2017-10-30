import tensorflow as tf
from My_Data.TF_Reader import TF_Reader
from My_Data.Data_Argument import *
from My_Trainer import Trainer
from Attention_Model import Attention_Model
from tensorflow.python.platform import flags
import common_flags
import My_Log.Log_Manager as LM
import tensorflow.contrib.slim as slim
import sys
import os
import cv2
import numpy as np

net_img = tf.placeholder(tf.uint8, [70,200,3])
img = tf.reshape(net_img,[-1,3])
# img = tf.reshape(img,[70,200,3])

real_img_path = "/home/night/data/cap2/test/14_tcSkUGuF.jpg"
real_img = cv2.imread(real_img_path)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
res_img = sess.run(img,feed_dict={net_img:real_img})

res_img = np.reshape(res_img,[70,200,3])
cv2.imshow("0",real_img)
cv2.imshow("1",res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()