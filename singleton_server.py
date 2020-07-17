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
from flask import Flask, flash, request, redirect, url_for, send_from_directory,g
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from flask_cors import CORS, cross_origin
import json

UPLOAD_FOLDER = './server_audio'
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

def detect_leading_silence(sound, silence_threshold=-5.0, chunk_size=10):
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def trim_silence_add_pass(path, exportPath):
    sound = AudioSegment.from_file(path, format="wav")
    silence_threshold = sound.dBFS - 10

    duration = len(sound)    

    start_trim = detect_leading_silence(sound,silence_threshold) 
    end_trim = detect_leading_silence(sound.reverse(),silence_threshold)

    if (start_trim > 100):
        start_trim = start_trim - 50
    if (duration - end_trim > 200):
        end_trim = end_trim - 50
    else:
        end_trim = 0

    trimmed_sound = sound[start_trim:duration-end_trim]
    # trimmed_sound = trimmed_sound.low_pass_filter(2000)

    trimmed_sound.export(exportPath,format = "wav")  

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
            return  json.dumps(text, ensure_ascii=False)
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
    
def initialize_model():
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    sess = tf.Session()
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
    
    return sess


def get_model():
    if 'model' not in g:
        g.model = initialize_model()
    
    return g.model

with app.app_context():
    load_sess = get_model()







def getVoiceToText():

    inputs = tf.placeholder(tf.float32,shape=(None, None, model_config["n_input_fetures"]),name="inputs")
    labels = tf.placeholder(tf.int32,shape=(None, None),name='labels')
    
    label_lengths = tf.placeholder(tf.int32, shape=(None))
    input_lengths = tf.placeholder(tf.int32, shape=(None))

    deep_speech_model = model(inputs, input_lengths, labels,
                            label_lengths, model_config, 0.95, mode=ModelMode.TEST)
    
    # trim_silence_add_pass("./server_audio/data.wav","./server_audio/data_edit.wav")

    audio_input = [featurize("./server_audio/data.wav")]
    audio_input_length = [np.shape(audio_input)[1]]

    
    # print(audio_input_length)
    l, s = load_sess.run(deep_speech_model, feed_dict={
        inputs: audio_input, input_lengths: audio_input_length})

    decode = batch_decode(l, s)
    result = list_char_to_string(decode[0])

    return result


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(host='0.0.0.0',debug=True, port=8000,ssl_context='adhoc')

   
