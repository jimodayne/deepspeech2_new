import os
import sys
import math
import argparse
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from model import model, ModelMode, model_config
from decoder import batch_decode, batch_label_to_text, list_char_to_string
from nltk.metrics import distance

def compute_cer(predict, label) :
    """
        Dùng để tính character error rate
        predict :   string dự đoán được từ mô hình
        label   :   string true label

        chương trình trả về character error rate
    """
    return distance.edit_distance(predict, label)/len(label)

def compute_wer(predict, label) :
    """
        Dùng để tính character error rate
        predict :   string dự đoán được từ mô hình
        label   :   string true label

        chương trình trả về word error rate
    """
    words = set(predict.split() + label.split())
    word2char = dict(zip(words, range(len(words))))

    new_predict = [chr(word2char[w]) for w in predict.split()]
    new_label = [chr(word2char[w]) for w in label.split()]
    label_size = np.size(new_label)

    return distance.edit_distance(''.join(new_predict), ''.join(new_label))/label_size

def main(check_point_directory, test_json_file, result_file):
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
    deep_speech_model = model(inputs, input_lengths, labels, label_lengths, model_config,0.95, mode=ModelMode.TEST)
     
    # print(deep_speech_model.mode)
    # print(deep_speech_model.config["n_input_fetures"])
    ################################

    # using data generator
    datagen = DataGenerator()
    datagen.load_train_data(test_json_file)
    # datagen.load_train_data("./train_json.json")

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
            print("-----------------/////////------------------")
            return

        dataset = datagen.iterate_train()
        logits, sequence_lengths = [], []
        wer, cer = [], []
        # sử dụng file để lưu lại các kết quả dự đoán của mô hình
        f = open(result_file, "w")
        
        for i , batch in enumerate(dataset):
            # print("batch['x']:", np.array(batch['x']))
            # print(" input_lengths:",  batch['input_lengths'])
            l, s = sess.run( deep_speech_model, 
                                        feed_dict={inputs: batch['x'], 
                                                    # labels : batch['y'],
                                                    # label_lengths : batch['label_lengths'],
                                                    input_lengths : batch['input_lengths']
                                                    }
                                        )
            logits.append(l)
            sequence_lengths.append(s)
            print("shape of logits batch {}".format(i + 1), np.shape(l))
            print("sequence lengths", s)
            label = batch_label_to_text(batch['y'], batch['label_lengths'])
            decode = batch_decode(l, s)
            for count in range(np.size(s)) :
                true_label  = list_char_to_string(label[count])
                predict     = list_char_to_string(decode[count])
                f.write("True label :" + true_label)
                print("True label :" + true_label)
                f.write("\n")
                f.write("Predict    :" + predict)
                print("Predict    :" + predict)
                f.write("\n")
                f.write("\n")
                c = compute_cer(predict, true_label)
                w = compute_wer(predict, true_label)
                cer.append(c)
                wer.append(w)
                print("cer :", c)
                print("wer :", w)
            # break
        average_cer = np.mean(cer)
        average_wer = np.mean(wer)
        # print(average_cer)
        # print(average_wer)
        f.write("\n average character error rate"   + str(average_cer))
        f.write("\n average word error rate"        + str(average_wer))
if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('check_point_directory', type=str, 
                        help='Path to data directory') 
    parser.add_argument('test_json_file', type=str, 
                        help='Path to test file') 
    parser.add_argument('result_file', type=str, 
                        help='Path to result file') 
    args = parser.parse_args() 
    main(args.check_point_directory, args.test_json_file, args.result_file)