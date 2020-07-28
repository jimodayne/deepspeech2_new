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
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
import io

UPLOAD_FOLDER = './server_audio'
LM_DIRECTORY = './check_point_cse/word_model_left.model'
check_point_directory = "./check_point_cse"

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


def sample_recognize(local_file_path='./server_audio/data_edit.wav'):
    """
    Transcribe a short audio file using synchronous speech recognition
    Args:
      local_file_path Path to local audio file, e.g. /path/audio.wav
    """

    client = speech_v1.SpeechClient()

    # local_file_path = 'resources/brooklyn_bridge.raw'

    # The language of the supplied audio
    language_code = "vi-VN"

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 16000

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.

    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
    }
    with io.open(local_file_path, "rb") as f:
        content = f.read()
    audio = {"content": content}

    response = client.recognize(config, audio)
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]

        print(u"Transcript: {}".format(alternative.transcript))
    return response


def featurize(audio_clip, step=10, window=20, max_freq=22050, desc_file=None):
    return spectrogram_from_file(
        audio_clip, mode=ModelMode.TEST, step=step, window=window,
        max_freq=max_freq)


def detect_leading_silence(sound, silence_threshold=-5.0, chunk_size=10):
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def match_target_amplitude(path, exportPath, target_dBFS):
    sound = AudioSegment.from_file(path, format="wav")

    processed_sound.export(exportPath, format="wav")


def trim_silence_add_pass(path, exportPath):
    target_dBFS = -20
    sound = AudioSegment.from_file(path, format="wav")
    silence_threshold = -30

    duration = len(sound)

    start_trim = detect_leading_silence(sound, silence_threshold)
    end_trim = detect_leading_silence(sound.reverse(), silence_threshold)

    if (start_trim > 100):
        start_trim = start_trim - 100
    else:
        start_trim = 0

    if (duration - end_trim > 200):
        end_trim = end_trim - 100
    else:
        end_trim = 0

    trimmed_sound = sound[start_trim:duration-end_trim]

    change_in_dBFS = target_dBFS - trimmed_sound.dBFS
    processed_sound = trimmed_sound.apply_gain(change_in_dBFS)
    # trimmed_sound = trimmed_sound.low_pass_filter(2000)

    processed_sound.export(exportPath, format="wav")


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

            googleApi_res = sample_recognize()

            text = getVoiceToText()
            model_w2v = Word2Vec.load(LM_DIRECTORY)
            correct_lm = correct_by_word(text, model_w2v)
            # return  json.dumps({text,correct_lm}, ensure_ascii=False)
            return jsonify(predict=text, correct=correct_lm, googleApi=googleApi_res)
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
                    list_candidate_word = word_vectors.most_similar(
                        [text_list[i-1], text_list[i+1]], topn=400)
                    check = 1
            elif i < len(text_list) - 1 and text_list[i+1] in alter_list:
                list_candidate_word = word_vectors.most_similar(
                    text_list[i+1], topn=400)
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

        trim_silence_add_pass("./server_audio/data.wav",
                              "./server_audio/data_edit.wav")

        audio_input = [featurize("./server_audio/data_edit.wav")]
        audio_input_length = [np.shape(audio_input)[1]]

        # print(audio_input_length)
        l, s = sess.run(deep_speech_model, feed_dict={
            inputs: audio_input, input_lengths: audio_input_length})

        decode = batch_decode(l, s)
        result = list_char_to_string(decode[0])

        return result


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(host='0.0.0.0', debug=True, port=8001, ssl_context='adhoc')
