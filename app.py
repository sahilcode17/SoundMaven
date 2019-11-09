from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import librosa

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# define a flask app
app = Flask(__name__)

model = load_model('final_noisy')

print("Model loaded. Start serving")

with open("lab_noisy.txt", 'r') as f:
    lab = [line.rstrip('\n') for line in f]

    
def getPredictedLabel(test_X):
    first_image_id = 0
    return sorted(set(lab))[np.argmax(model.predict(test_X.reshape(1, -1))[first_image_id])]

file_name = "/home/chutiya/Documents/Project/input/0a0c99af.wav"
#C:/Users/Kartikay Bansal/Project III Sem/input/freesound-audio-tagging-2019
test_X, sample_rate = librosa.load(file_name,res_type='kaiser_fast')
mfccs = np.mean(librosa.feature.mfcc(y=test_X, sr=sample_rate, n_mfcc=40).T,axis=0)
test_X = np.array(mfccs)
getPredictedLabel(test_X)
print(getPredictedLabel(test_X))



@app.route('/', methods=['GET'])
def index():
   
   return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
   
    prediction = getPredictedLabel(test_X)

    return render_template("index.html",prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True , port="171998")