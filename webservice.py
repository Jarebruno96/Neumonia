import sys
import os
import json
import numpy as np
import cv2
from PIL import Image
import jsonpickle
import getopt
from flask import Flask, request, Response
from keras.models import load_model
from chestxrayprocessor import chestxrayprocessor
import tensorflow as tf
from mypprint import mypprint

app = Flask(__name__)

model = None
processorOpts = None

def usage():
    mypprint.printBlue("python webservice.py -f|--folder folder")
    mypprint.printBlue("\t -f --folder specifies model´s folder to be loaded")

def main(argv):

    global model
    global processorOpts

    # Reading and parsing args
    try:
        opts, args = getopt.getopt(argv, 
                                    "hf:", 
                                    ["help", 
                                     "folder="]
                                    )
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)
    
    modelFolder = None

    for opt, value in opts:
        if opt in ("-f", "--folder"):
            modelFolder = value
    
    if modelFolder is None:
        mypprint.printError("Model Folder is not defined")
        usage()
        sys.exit(2)
    
    # Loading model

    model = load_model(os.path.join(modelFolder, "model.h5"))
    model._make_predict_function()

    with open(os.path.join(modelFolder, "parametersSummary.json")) as fp:
        conf = json.load(fp)

    processorOpts = chestxrayprocessor.ProcessorOpts
    processorOpts.channels = conf["Channels"]
    processorOpts.height = conf["Image height"]
    processorOpts.width = conf["Image width"]
    processorOpts.normalize = conf["Normalize"]

    # Run flask app
    app.run()

@app.route("/status")
def hello_world():
    return "OK"


@app.route("/predict", methods=["POST"])
def predict():

    global model
    global processorOpts

    r = request
    nparr = np.fromstring(r.data, np.uint8)
    
    img = np.array(Image.open(r.files['file']))

    # Check if image is in RGB scale
    if processorOpts.channels == 1:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRc)

    imgProccesed = chestxrayprocessor.processImage(img, processorOpts)

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

    # Encode response using jsonpickle

    response_pickled = jsonpickle.encode(response)
    resp = Response(response=response_pickled, status=200, mimetype="application/json")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    main(sys.argv[1:])
