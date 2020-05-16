import os
import sys
import math
import argparse
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from model import model, ModelMode, model_config
from decoder import batch_decode, batch_label_to_text, list_char_to_string, compute_cer, compute_wer
from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence
import socket


serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

serv.bind(('0.0.0.0', 8080))

# serv.bind(('localhost', 8888))
serv.listen(5)  

def featurize(audio_clip, step=10, window=20, max_freq=22050, desc_file=None) :
    return spectrogram_from_file(
            audio_clip,mode=ModelMode.TEST, step=step, window=window,
            max_freq=max_freq)

def main(check_point_directory="./check_point"):
    # input là một batch của dataset, ở đây có sự khác nhau lúc train và inference về số batch size
    # shape của input là các tham số lần lượt là: shape=[batch_size, max_time_steps, num_features]
    inputs = tf.placeholder(
            tf.float32, 
            shape=(None, None, model_config["n_input_fetures"]), 
            name="inputs")

  
    # labels
    labels = tf.placeholder(
            tf.int32, 
            shape=(None, None), 
            name='labels')

   
    label_lengths = tf.placeholder(tf.int32, shape=(None))

    input_lengths = tf.placeholder(tf.int32, shape=(None))
    

    
    # create an instance of model
    # deep_speech_model = model(inputs, input_lengths, labels, label_lengths, model_config, mode=ModelMode.TEST)
    deep_speech_model = model(inputs, input_lengths, labels, label_lengths, model_config,0.95, mode=ModelMode.TEST)
    

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess :
        sess.run(init_op)
        try :
            saver.restore(sess, tf.train.latest_checkpoint(check_point_directory))
            # saver.restore(sess, tf.train.latest_checkpoint('./check_point/'))
            print(" ")
            print("restore check point success")
            print("-----------------/////////------------------")
        except : 
            print(" ")
            print("can not find check point at ", check_point_directory)
            print("-----------------////=/////------------------")
            # return
        
        # create a audio for inference
      
        conn, addr = serv.accept()     # Establish connection with client.
        print('Got connection from', addr)
        f = open('./server_audio/data.wav','ab')
        
        while (True):       
        # receive data and write it to file
            data = conn.recv(1024)
            if not data: 
                break
            f.write(data)
        
            
        print("Done Receiving")
        
        f.close()

        audio_input = [featurize("./server_audio/data.wav")]

        # print("audio_input:", np.shape(audio_input))

        # audio_input_length = [np.shape(audio_input)[1]]

       
        # input_lengths = [f.shape[0] for f in features]
        # max_length = max(input_lengths)
       

        audio_input_length = [np.shape(audio_input)[1]]
        
    
        print(audio_input_length)


        l, s = sess.run( deep_speech_model, feed_dict={inputs : audio_input, input_lengths : audio_input_length})
    


        decode = batch_decode(l, s)
        result = list_char_to_string(decode[0])
        print("result", result)

        # serv.listen(5)  
        conn.send(result.encode())
        print("close connection from", addr)
        conn.close()                # Close the connection

main()
