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
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from flask_cors import CORS, cross_origin
import editdistance
from gensim.models import Word2Vec
from flask import jsonify


UPLOAD_FOLDER = './server_audio'
LM_DIRECTORY = './check_point_cse/word_model_left.model'
check_point_directory = "./check_point_cse"

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


def featurize(audio_clip, step=10, window=20, max_freq=22050, desc_file=None):
    return spectrogram_from_file(
        audio_clip, mode=ModelMode.TEST, step=step, window=window,
        max_freq=max_freq)



@cross_origin()
@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = "data.wav"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            text = getVoiceToText()
            model_w2v = Word2Vec.load(LM_DIRECTORY)
            correct_lm = correct_by_word(text,model_w2v)
            # return  json.dumps({text,correct_lm}, ensure_ascii=False)
            return jsonify(predict = text, correct = correct_lm)
            # return redirect(url_for("getVoiceToText"))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''



def correct_by_word(TEXT, model_w2v):
  text = TEXT.lower()
  text_list = text.split()

  word_vectors = model_w2v.wv
  alter_list = list(word_vectors.vocab.keys())

  for i in range(len(text_list)):
    if text_list[i] not in alter_list:
      min_distance = 100
      alter_word = ""
      check = 0
      if i > 0 and i < len(text_list) - 1:
        if text_list[i-1] in alter_list and text_list[i+1] in alter_list:
          list_candidate_word = word_vectors.most_similar([text_list[i-1], text_list[i+1]], topn=400)
          check = 1
      elif i < len(text_list) - 1 and text_list[i+1] in alter_list:
        list_candidate_word = word_vectors.most_similar(text_list[i+1], topn=400)
        check = 1

      if check == 1:
        for word in list_candidate_word:
          if editdistance.eval(word[0], text_list[i]) < min_distance:
            min_distance = editdistance.eval(word[0], text_list[i])
            alter_word = word[0]
        text_list[i] = alter_word
      else:
        for word in alter_list:
          if editdistance.eval(word, text_list[i]) < min_distance:
            min_distance = editdistance.eval(word, text_list[i])
            alter_word = word
        text_list[i] = alter_word

  return " ".join(text_list)


def getVoiceToText():
    inputs = tf.placeholder(
        tf.float32,
        shape=(None, None, model_config["n_input_fetures"]),
        name="inputs")

    labels = tf.placeholder(
        tf.int32,
        shape=(None, None),
        name='labels')

    label_lengths = tf.placeholder(tf.int32, shape=(None))
    input_lengths = tf.placeholder(tf.int32, shape=(None))
    deep_speech_model = model(inputs, input_lengths, labels,
                              label_lengths, model_config, 0.95, mode=ModelMode.TEST)
    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        try:
            saver.restore(sess, tf.train.latest_checkpoint(
                check_point_directory))
            print(" ")
            print("restore check point success")
            print("-----------------/////////------------------")
        except:
            print(" ")
            print("can not find check point at ", check_point_directory)
            print("-----------------////=/////------------------")

    

        audio_input = [featurize("./server_audio/data.wav")]
        audio_input_length = [np.shape(audio_input)[1]]

        # print(audio_input_length)
        l, s = sess.run(deep_speech_model, feed_dict={
            inputs: audio_input, input_lengths: audio_input_length})

        decode = batch_decode(l, s)
        result = list_char_to_string(decode[0])

        return result


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(host='0.0.0.0',debug=True, port=8000,ssl_context='adhoc')
