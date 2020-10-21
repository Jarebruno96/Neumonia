import os
from mypprint import mypprint 
import cv2
import numpy as np
from google.cloud import storage


GCP_SOURCE = "gcp"
LOCAL_SOURCE = "local"
GRAY_SCALE_CHANNELS = 1
JPEG_CONTENT_TYPE = "image/jpeg"

class RetrieverOpts:
    dataFolder = "chest_xray"
    trainFolder = "train"
    testFolder = "test"
    valFolder = "val"
    normalFolder = "NORMAL"
    pneumoniaFolder = "PNEUMONIA"
    source = "local"
    googleStorageKey = ""
    bucket = ""


def getTrainData(retrieverOpts, processorOpts):
    return getDataFrom(retrieverOpts, processorOpts, retrieverOpts.trainFolder)

def getTestData(retrieverOpts, processorOpts):
    return getDataFrom(retrieverOpts, processorOpts, retrieverOpts.testFolder)

def getDataFrom(retrieverOpts, processorOpts, dataType):

    if retrieverOpts.source == GCP_SOURCE:
        return getDataFromGCP(dataType, processorOpts, retrieverOpts)
    elif retrieverOpts.source == LOCAL_SOURCE:
        return getDataFromLocalDisk(dataType, processorOpts, retrieverOpts)
    else:
        mypprint.printError("Source " + source + " not known. Can not retrieve data")

    return None

def getDataFromGCP(dataType, processorOpts, retrieverOpts):

    x = []
    y = []

    client = None

    if not retrieverOpts.googleStorageKey:
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(retrieverOpts.googleStorageKey)
    
    bucket = client.bucket(retrieverOpts.bucket)

    blobs = bucket.list_blobs(prefix = retrieverOpts.dataFolder + "/" + dataType + "/" + retrieverOpts.normalFolder)

    for blob in blobs:
        if blob.content_type == JPEG_CONTENT_TYPE:

            if processorOpts.channels == GRAY_SCALE_CHANNELS:
                image = np.asarray(bytearray(blob.download_as_string()), dtype="uint8")
                x.append(np.array(np.expand_dims(cv2.imdecode(image, cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imdecode(image)))
            y.append(0)
    

    blobs = bucket.list_blobs(prefix = retrieverOpts.dataFolder + "/" + dataType + "/" + retrieverOpts.pneumoniaFolder)

    for blob in blobs:
        if blob.content_type == JPEG_CONTENT_TYPE:

            if processorOpts.channels == GRAY_SCALE_CHANNELS:
                image = np.asarray(bytearray(blob.download_as_string()), dtype="uint8")
                x.append(np.array(np.expand_dims(cv2.imdecode(image, cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imdecode(image)))
            y.append(1)
    
    return (np.array(x), np.array(y))



def getDataFromLocalDisk(dataType, processorOpts, retrieverOpts):

    x = []
    y = []

    path = os.path.join(retrieverOpts.dataFolder, dataType, retrieverOpts.normalFolder)
    filesNames = os.listdir(path)

    for fileName in filesNames:
        if fileName.endswith(".jpeg"):
            if processorOpts.channels == GRAY_SCALE_CHANNELS:
                x.append(np.array(np.expand_dims(cv2.imread(os.path.join(path, fileName), cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imread(os.path.join(path, fileName))))
            y.append(0)

    path = os.path.join(retrieverOpts.dataFolder, dataType, retrieverOpts.pneumoniaFolder)
    filesNames = os.listdir(path)

    for fileName in filesNames:

        if fileName.endswith(".jpeg"):
            if processorOpts.channels == GRAY_SCALE_CHANNELS:
                x.append(np.array(np.expand_dims(cv2.imread(os.path.join(path, fileName), cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imread(os.path.join(path, fileName))))
            y.append(1)

    return (np.array(x), np.array(y))