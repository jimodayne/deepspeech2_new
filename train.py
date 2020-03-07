import os
import sys
import math
import argparse
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from model import model, ModelMode, model_config
from decoder import batch_decode, batch_label_to_text, list_char_to_string, compute_cer, compute_wer
import csv
from tensorflow.core.framework import summary_pb2
import time
import random

def write_log_file(file_name,
                   num_epoch,
                   average_train_cost,
                   average_test_cost,
                   average_train_wer,
                   average_train_cer,
                   average_test_wer,
                   average_test_cer):
    with open(file_name, 'a') as f:
        f.write("{},{},{},{},{},{},{}\n".format(num_epoch,
                                                average_train_cost,
                                                average_test_cost,
                                                average_train_wer,
                                                average_train_cer,
                                                average_test_wer,
                                                average_test_cer))


def main(check_point_directory, train_json_file, test_json_file, log_file):
    global a
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
    keep_prop = tf.placeholder(tf.float32)

    # create an instance of model
    logits, sequence_lengths, cost, optimizer = model(
        inputs, input_lengths, labels, label_lengths, model_config, keep_prop, mode=ModelMode.TRAIN)

    # global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    increment_global_step = tf.assign_add(global_step, 1,
                                          name='increment_global_step')

    # using data generator, load data form train file
    train_datagen = DataGenerator()
    train_datagen.load_train_data(train_json_file)

    # using data generator, load data from test file
    test_datagen = DataGenerator()
    test_datagen.load_test_data(test_json_file)

    # compute time
    start_time = time.time()
    end_time = start = time.time()

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        train_writer = tf.summary.FileWriter('logs/train/', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test/', sess.graph)
        try:
            # saver.restore(sess, tf.train.latest_checkpoint('/content/drive/My Drive/DeepSpeech/check_point/'))
            saver.restore(sess, tf.train.latest_checkpoint(
                check_point_directory))
            print(" ")
            print("restore check point success")
            print("-----------------/////////------------------")
        except:
            print(" ")
            print("can not find check point at ", check_point_directory)
            print("run from start")
            print("-----------------/////////------------------")

        for m in range(251):
            # increase global step
            current_global_step = sess.run(increment_global_step)

            # add random seed for augmentation , random seed = current global step
            random.seed(current_global_step)

            # prepare for training
            train_dataset = train_datagen.iterate_train()
            test_dataset = test_datagen.iterate_test()
            train_cost, test_cost = [], []
            train_wer, train_cer, test_wer, test_cer = [], [], [], []

            # vòng for đầu tiên dùng để chạy training trên tập dữ liệu training
            # vòng for thứ hai để tính toán trên tập dữ liệu test
            # tại mỗi epoch sẽ tính wer, cer của tập test và train (tính trung bình)

            print('\n')
            print("begining of epoch {}".format(current_global_step))
            print('\n')

            for i, batch in enumerate(train_dataset):
                l, s, c, _ = sess.run([logits, sequence_lengths, cost, optimizer],
                                      feed_dict={inputs: batch['x'],
                                                 labels: batch['y'],
                                                 label_lengths: batch['label_lengths'],
                                                 input_lengths: batch['input_lengths'],
                                                 keep_prop: 0.8}
                                      )

                train_cost.append(c)

                label = batch_label_to_text(batch['y'], batch['label_lengths'])
                decode = batch_decode(l, s)

                for count in range(np.size(s)):
                    true_label = list_char_to_string(label[count])
                    predict = list_char_to_string(decode[count])
                    cer = compute_cer(predict, true_label)
                    wer = compute_wer(predict, true_label)
                    train_cer.append(cer)
                    train_wer.append(wer)

                if i % 10 == 0:
                    end_time = time.time()
                    execute_time = end_time - start_time
                    start_time = time.time()
                    print("Trainging cost: epoch {}, batch {} has cost {} --- execution time {}".format(
                        current_global_step, i, c, execute_time))

            average_train_cer = np.mean(train_cer)
            average_train_wer = np.mean(train_wer)
            average_train_cost = np.mean(train_cost)

            for i, batch in enumerate(test_dataset):

                l, s, c = sess.run([logits, sequence_lengths, cost],
                                   feed_dict={inputs: batch['x'],
                                              labels: batch['y'],
                                              label_lengths: batch['label_lengths'],
                                              input_lengths: batch['input_lengths'],
                                              keep_prop: 1}
                                   )

                test_cost.append(c)

                label = batch_label_to_text(batch['y'], batch['label_lengths'])
                decode = batch_decode(l, s)
                for count in range(np.size(s)):
                    true_label = list_char_to_string(label[count])
                    predict = list_char_to_string(decode[count])
                    cer = compute_cer(predict, true_label)
                    wer = compute_wer(predict, true_label)
                    test_cer.append(cer)
                    test_wer.append(wer)

            average_test_cer = np.mean(test_cer)
            average_test_wer = np.mean(test_wer)
            average_test_cost = np.mean(test_cost)

            write_log_file(log_file,
                           current_global_step,
                           average_train_cost,
                           average_test_cost,
                           average_train_wer,
                           average_train_cer,
                           average_test_wer,
                           average_test_cer)

            if (m + 1) % 1 == 0:
                saver.save(sess, check_point_directory +
                           "/deep_speech", global_step=current_global_step)
                print("save check point tại epoch {}".format(current_global_step))
                
                tf_train_cost = summary_pb2.Summary.Value(
                    tag="average_cost", simple_value=average_train_cost)
                tf_train_wer = summary_pb2.Summary.Value(
                    tag="average_wer", simple_value=average_train_wer)
                tf_train_cer = summary_pb2.Summary.Value(
                    tag="average_cer", simple_value=average_train_cer)    
                train_summary = summary_pb2.Summary(value=[tf_train_cost, tf_train_wer, tf_train_cer])
                train_writer.add_summary(train_summary, current_global_step)
                train_writer.flush()

                tf_test_cost = summary_pb2.Summary.Value(
                    tag="average_cost", simple_value=average_test_cost)
                tf_test_wer = summary_pb2.Summary.Value(
                    tag="average_wer", simple_value=average_test_wer)
                tf_test_cer = summary_pb2.Summary.Value(
                    tag="average_cer", simple_value=average_test_cer)    
                test_summary = summary_pb2.Summary(value=[tf_test_cost, tf_test_wer, tf_test_cer])
                test_writer.add_summary(test_summary, current_global_step)
                test_writer.flush()

            print('\n')
            print("average_train_cer {}".format(average_train_cer))
            print("average_train_wer {}".format(average_train_wer))
            print("average_test_cer  {}".format(average_test_cer))
            print("average_test_wer  {}".format(average_test_wer))
            print("epoch {} train cost : ".format(
                current_global_step), average_train_cost)
            print("epoch {} test cost :  ".format(
                current_global_step), average_test_cost)
            print('\n')
            print("end of epoch {}".format(current_global_step))
            print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('check_point_directory', type=str,
                        help='Path to checkpoint directory')
    parser.add_argument('train_json_file', type=str,
                        help='Path to train file')
    parser.add_argument('test_json_file', type=str,
                        help='Path to test file')
    parser.add_argument('log_file', type=str,
                        help='Path to log file')
    args = parser.parse_args()
    main(args.check_point_directory, args.train_json_file,
         args.test_json_file, args.log_file)
