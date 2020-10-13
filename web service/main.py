from flask import Flask, request, Response
from keras.models import load_model
import os
import json
import sys
sys.path.append("../")
from chestxrayprocessor import chestxrayprocessor
from pprint import pprint
import numpy as np
import cv2
import jsonpickle
import tensorflow as tf
from PIL import Image

modelFolder = "../03-10-20_saves"

model = load_model(os.path.join(modelFolder, "model.h5"))
model._make_predict_function()
graph = tf.get_default_graph()


with open(os.path.join(modelFolder, "parametersSummary.json")) as fp:
    conf = json.load(fp)

processorOpts = chestxrayprocessor.ProcessorOpts
processorOpts.channels = conf["Channels"]
processorOpts.height = conf["Image height"]
processorOpts.width = conf["Image width"]
processorOpts.normalize = conf["Normalize"]

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello world"


@app.route("/predict", methods=["POST"])
def predict():

    global model
    global processorOpts
    global graph

    r = request
    nparr = np.fromstring(r.data, np.uint8)

    #if processorOpts.channels == 1:
    #    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    #else:
    #    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = np.array(Image.open(r.files['file']))

    imgProccesed = chestxrayprocessor.processImage(img, processorOpts)

    with graph.as_default():
        prediction = model.predict(np.array([imgProccesed]))
    
    prediction = round(prediction[0][0], 2)

    predictionResult = "Pneumonia"
    if prediction <= 0.5:
        predictionResult = "Normal"

    response = {
        'original': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
        'proccessed': 'image received. size={}x{}'.format(imgProccesed.shape[1], imgProccesed.shape[0]),
        'prediction' : str(prediction),
        'predictionResult' : predictionResult

    }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    resp = Response(response=response_pickled, status=200, mimetype="application/json")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp