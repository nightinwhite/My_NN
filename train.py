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

common_flags.define()
Flages = flags.FLAGS

def single_accuracy(net_predict,net_lable):
    single_acc = tf.cast(tf.equal(tf.argmax(net_predict, -1), tf.argmax(net_lable, -1)), tf.float32)
    single_acc = tf.reduce_mean(single_acc)
    return single_acc

def seq_accuracy(net_predict,net_lable):
    seq_acc = tf.cast(tf.equal(tf.argmax(net_predict, -1), tf.argmax(net_lable, -1)), tf.float32)
    seq_acc = tf.reduce_sum(seq_acc, -1)
    seq_acc = tf.cast(tf.equal(seq_acc, Flages.label_length), tf.float32)
    seq_acc = tf.reduce_mean(seq_acc)
    return seq_acc

def get_loss(pred_res,net_label):
    with tf.variable_scope('sequence_loss'):
        labels_list = tf.unstack(label_smoothing_regularization(net_label), axis=1)
        batch_size, seq_length, _ = net_label.shape.as_list()
        weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
        logits_list = tf.unstack(pred_res, axis=1)
        weights_list = tf.unstack(weights, axis=1)
        loss = tf.contrib.legacy_seq2seq.sequence_loss(
            logits_list,
            labels_list,
            weights_list,
            softmax_loss_function=tf.nn.softmax_cross_entropy_with_logits,
            average_across_timesteps=False  # !!!!!
        )
        regularizer_loss = tf.losses.get_regularization_loss()
        return loss + regularizer_loss
def label_smoothing_regularization(chars_labels, weight=0.1):
        # print type(FLAGS.num_char_classes)
        # print FLAGS.num_char_classes
        pos_weight = 1.0 - weight
        neg_weight = weight / Flages.number_of_class
        return chars_labels * pos_weight + neg_weight

arg_ops = [
    base_norm,
    # distorting_color
]

# tmp_path = "/media/night/0002FCA800053168/data/fsns_tf_train/"
# tmp_files_name = os.listdir(tmp_path)
# train_files_name = []
# for name in tmp_files_name:
#     train_files_name.append(os.path.join(tmp_path,name))
#
# tmp_path = "/media/night/0002FCA800053168/data/fsns_tf_validation/"
# tmp_files_name = os.listdir(tmp_path)
# val_files_name = []
# for name in tmp_files_name:
#     val_files_name.append(os.path.join(tmp_path,name))


train_tf_reader = TF_Reader(["tfrecords/chinese1_train_small.record"],True)
train_tf_reader.data_argument(arg_ops)
train_images,train_labels = train_tf_reader.shuffle_batch()
val_tf_reader = TF_Reader(["tfrecords/chinese1_val.record"],True)
val_tf_reader.data_argument(arg_ops)
val_images,val_labels = val_tf_reader.shuffle_batch()

train_model = Attention_Model(train_images,train_labels)
train_model.build_model("Attention_Model")
train_model_pred_res = train_model.pred_res

val_model = Attention_Model(val_images)
val_model.is_training = False
val_model.build_model("Attention_Model",True)
val_model_pred_res = val_model.pred_res

train_loss = get_loss(train_model_pred_res,train_labels)
val_loss = get_loss(val_model_pred_res,val_labels)

train_single_acc = single_accuracy(train_model_pred_res,train_labels)
train_seq_acc = seq_accuracy(train_model_pred_res,train_labels)
val_single_acc = single_accuracy(val_model_pred_res,val_labels)
val_seq_acc = seq_accuracy(val_model_pred_res,val_labels)

epoch = Flages.epoch
iteration_in_epoch  = Flages.train_iter_num
iteration_in_val = Flages.val_iter_num

sess = tf.InteractiveSession()

# summarys:
all_variables = tf.global_variables()
for v in all_variables:
    tf.summary.histogram(v.name,v)
tf.summary.image("train_images",train_images,10)
tf.summary.scalar("train_loss",train_loss)
tf.summary.scalar("train_single_acc",train_single_acc)
tf.summary.scalar("train_seq_acc",train_seq_acc)
merge_summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("train_log/",sess.graph)

my_trainer = Trainer.Trainer(sess = sess,
                             train_loss = train_loss,
                             optimizer_name = 'MomentumOptimizer',
                             learning_rate = Flages.learning_rate,
                             epoch = Flages.epoch,
                             train_iter_num = Flages.train_iter_num,
                             val_iter_num = Flages.val_iter_num
                             )
my_trainer.set_clip_value(Flages.clip_gradient_norm)

train_info_ops = [
    Trainer.Info_Ops(train_single_acc,"train_single_acc",1),
    Trainer.Info_Ops(train_seq_acc,"train_seq_acc",1),
]
my_trainer.set_train_info_ops(train_info_ops)

val_info_ops = [
    Trainer.Info_Ops(val_loss,"val_loss",1),
    Trainer.Info_Ops(val_single_acc,"val_single_acc",1),
    Trainer.Info_Ops(val_seq_acc, "val_seq_acc",1),
]
my_trainer.set_val_info_ops(val_info_ops)

my_trainer.set_save_info("models/","attention_model_chinese1_with_cap2")

def cmp_value(bef_value, now_value):
    if bef_value < now_value:
        return True
    else:
        return False

my_trainer.set_best_save_info("val_seq_acc",cmp_fuc=cmp_value)

my_trainer.set_summary_writer(merge_summary, summary_writer, 10)

my_trainer.init_model()

train_model.restore_cnn_model(sess, "models/attention_model_Momentum_val_seq_acc_0.93125")

my_trainer.add_visual_info("train_loss")
my_trainer.add_visual_info("train_single_acc")
my_trainer.add_visual_info("val_loss")
my_trainer.add_visual_info("val_single_acc")

my_trainer.run()
