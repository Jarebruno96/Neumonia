import sys
import numpy as np
from PIL import Image
import jsonpickle
from flask import Flask, request, Response
from mypprint import mypprint
from chestxrayprocessor import chestxrayprocessor
from osutils import osutils
from chestxrayio import chestxrayio


app = Flask(__name__)
model = None
processorOpts = None


def main(argv):

    global model
    global processorOpts

    # Reading and parsing args

    ioOpts, _, _ = osutils.parseArgs(argv)
    
    if ioOpts.modelFolder is None:
        mypprint.printError("Model Folder parameter is not defined")
        sys.exit(2)
    

    if ioOpts.source == chestxrayio.GCP_SOURCE:
        ioOpts.bucket = chestxrayio.getGCPStorageBucket(ioOpts)
    

    # Loading model
    mypprint.printText("Cargando modelo")
    model = chestxrayio.loadModel("model.h5", ioOpts)


    # Loading preprocesing images options

    processorOpts = chestxrayio.loadPreprocesingOpts("parametersSummary.json", ioOpts)


    # Run flask app

    app.run()


@app.route("/status")
def hello_world():

    return "OK"


@app.route("/predict", methods=["POST"])
def predict():

    global model
    global processorOpts

    # Read request iamge
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = np.array(Image.open(r.files['file']))
    imgProccesed = chestxrayprocessor.parseRequestImage(img, processorOpts)


    #Predict 
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
