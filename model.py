
from __future__ import absolute_import, division, print_function


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import math
import argparse
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
# from data import utils
import matplotlib.pyplot as plt


class ModelMode(object):
    TRAIN = 0
    TEST = 1
    INFER = 2

model_config = {
    "n_input_fetures" : 442,
    # config for CNN layers
    "n_cnn_layers" : 1,
    "num_classes" : 95,

    # layer 1
    "conv1_padding" : "SAME",
    "conv1_stride" : [2, 2], #h x w
    "conv1_kernel_shape": [5, 5], #h x w
    "conv1_in_chanels" : 1,
    "conv1_out_chanels" : 32,
    # "conv1_keep_prob" : 0.95, # keep_prop dropout

    # layer 2
    "conv2_padding" : "SAME",
    "conv2_stride" : [1, 1], #h x w
    "conv2_kernel_shape": [5, 5], #h x w
    "conv2_in_chanels" : 32,
    "conv2_out_chanels" : 32,
    # "conv2_keep_prob" : 0.95, # keep_prop dropout

    # layer 3
    # "conv3_padding" : "SAME",
    # "conv3_stride" : [2, 1], #h x w
    # "conv3_kernel_shape": [5, 5], #h x w
    # "conv3_in_chanels" : 8,
    # "conv3_out_chanels" : 8,
    # "conv3_keep_prob" : 0.95, # keep_prop dropout
    # rnn layer
}

deep_speech_model_mode = ModelMode.TRAIN

def conv_layer(inputs, config, name, cnn_keep_prop):
    # tạm thời sử dụng conv_1d theo giống mô hình deepspeech của baidu
    # có thể tìm thấy ở : https://github.com/baidu-research/ba-dls-deepspeech
    # có sự kết hợp của dropout tại conv layer này

        
    # tuy sử dụng là tf.nn.conv_2d nhưng thực chất là conv_1d
    
    conv_outputs = tf.layers.conv2d(
        inputs=inputs, 
        filters=config['conv{}_out_chanels'.format(name)],        # số lượng filters trong lớp conv
        kernel_size=config['conv{}_kernel_shape'.format(name)],   # conv window shape [height, width] 
        strides=config['conv{}_stride'.format(name)],             # mỗi bước stride là [height x width]
        padding=config['conv{}_padding'.format(name)],            # sử dụng padding
        use_bias=True, 
        # kernel_initializer=tf.random_normal_initializer,
        name="cnn_layer_{}".format(name)
    )
    conv_outputs = tf.nn.relu(conv_outputs)
    # conv_outputs = tf.clip_by_value(conv_outputs, 0, 20) # activation

    # conv_outputs = tf.keras.layers.BatchNormalization(epsilon=0.001)(conv_outputs) # batch normalize
    
    conv_outputs = tf.nn.dropout(
        conv_outputs, 
        keep_prob=cnn_keep_prop,   # dropout
        name="conv{}_dropout".format(name) 
    )  

    return conv_outputs

def rnn_layer( inputs, config, name, hidden_layer_size=1024, keep_prob=0.95, mode="LSTM", rnn_type="BI_DIR") :
        
    # input của rnn layer sẽ có dạng [batch_size, time_steps, num_features] (conv_outputs)

    def rnn_cell():
        cell = None
        if mode == 'LSTM':
            cell = tf.nn.rnn_cell.LSTMCell(hidden_layer_size, 
                                            name="rnn_{}".format(name))
        elif mode == 'GRU':
            cell = tf.nn.rnn_cell.GRUCell(hidden_layer_size, 
                                            name="rnn_{}".format(name))
        else:
            raise Exception('Invalid rnn type. It should be LSTM or GRU')

        return cell


    fw_rnn_cell = rnn_cell()
    bw_rnn_cell = rnn_cell()
            
    if rnn_type == 'BI_DIR' :
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                                            cell_fw=fw_rnn_cell, 
                                            cell_bw=bw_rnn_cell, 
                                            inputs=inputs,
                                            dtype=tf.float32,
                                            swap_memory=True)
        outputs_fw, outputs_bw = rnn_outputs
        rnn_outputs = outputs_fw + outputs_bw
    else :
        rnn_outputs, _ = tf.nn.dynamic_rnn(
                                            fw_rnn_cell, 
                                            inputs=inputs,
                                            dtype=tf.float32,
                                            swap_memory=True)

    rnn_outputs = tf.nn.relu(rnn_outputs)

    # rnn_outputs = tf.keras.layers.BatchNormalization(epsilon=0.001)(rnn_outputs)
    
    rnn_outputs = tf.nn.dropout(
        rnn_outputs, 
        keep_prob=keep_prob, 
        name="rnn{}_dropout".format(name)
    )
    return rnn_outputs
    
def fully_connected_layer(inputs, config):
    outputs = tf.layers.dense(
            inputs, 
            config['num_classes'], 
            activation=None, # keep linear
            # kernel_initializer=tf.random_normal_initializer,
            name='fully_connected_layer'
        )
    # outputs = tf.clip_by_value(outputs, 0, 20)
    return tf.transpose(outputs, [1, 0, 2])

def compute_cost(inputs, input_lengths, labels, label_lengths, sequence_length):

    # sau khi tinh toan xong ta tinh ctc loss cua moi batch, sau do cong lai lay trung binh
    sparse_label = tf.keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)
    losses = tf.nn.ctc_loss(sparse_label, inputs, sequence_length, ignore_longer_outputs_than_inputs=True)

    cost = tf.reduce_mean(losses)
    return cost

def training_optimizer(learning_rate=1e-5):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    return optimizer


def model(inputs, input_lengths, labels, label_lengths, config, cnn_keep_prop, mode=ModelMode.TRAIN):
    # sử dụng tf.expand_dims để tạo ra input thích hợp cho đầu vào của conv layer
    conv_outputs = tf.expand_dims(inputs, -1, name="cnn_input")    
    
    # conv layer 
    conv_outputs = conv_layer(conv_outputs, config, 1, cnn_keep_prop)
    conv_outputs = conv_layer(conv_outputs, config, 2, cnn_keep_prop)
    # conv_outputs = conv_layer(conv_outputs, config, 3)

    batch_size = tf.shape(conv_outputs)[0]
    num_features = conv_outputs.get_shape().as_list()[2]
    num_chanels = config["conv2_out_chanels"]
    conv_outputs = tf.reshape(                                                                                                                                          
                    conv_outputs,                                       
                    (batch_size, -1, num_features * num_chanels),                           
                    name="output_conv"
            )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

    ############################################
    rnn_outputs = rnn_layer(conv_outputs ,config, 1, hidden_layer_size=600, keep_prob=cnn_keep_prop, mode="LSTM", rnn_type="BI_DIR")

    rnn_outputs = rnn_layer(rnn_outputs ,config, 2, hidden_layer_size=600, keep_prob=cnn_keep_prop, mode="LSTM", rnn_type="BI_DIR")

    rnn_outputs = rnn_layer(rnn_outputs ,config, 3, hidden_layer_size=600, keep_prob=cnn_keep_prop, mode="LSTM", rnn_type="BI_DIR")

    # rnn_outputs = rnn_layer(rnn_outputs ,config, 4, hidden_layer_size=10, mode="LSTM", rnn_type="BI_DIR")

    # rnn_outputs = rnn_layer(rnn_outputs ,config, 5, hidden_layer_size=1000, mode="LSTM", rnn_type="BI_DIR")

    logits = fully_connected_layer(rnn_outputs, config)
    
    ##########################################################
    # tính toán sequence length cho việc tính toán cost và infer
    # hiện tại sử dụng max timestep, sau đó sẽ sửa lại sau, 
    # sẽ tính ra được những khoảng timestep không phải padding zero
    max_logits_time_steps = tf.shape(logits)[0]
    max_inputs_time_steps = tf.cast(tf.shape(inputs)[1], tf.float32)
    sequence_lengths = tf.cast(tf.multiply(max_logits_time_steps, input_lengths), tf.float32)
    sequence_lengths = tf.cast(tf.floordiv(sequence_lengths, max_inputs_time_steps), tf.int32)
    ##############################################################
    if mode != ModelMode.TRAIN :
        return [tf.transpose(logits, [1, 0, 2]), sequence_lengths]


    cost = compute_cost(logits, input_lengths, labels, label_lengths, sequence_lengths)

    optimizer = training_optimizer(2.e-5).minimize(cost)

    return [tf.transpose(logits, [1, 0, 2]), sequence_lengths, cost, optimizer]


  
