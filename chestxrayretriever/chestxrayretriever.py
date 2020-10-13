import os
from mypprint import mypprint 
import cv2
import numpy as np

class RetrieverOpts:
    dataFolder = "chest_xray"
    trainFolder = "train"
    testFolder = "test"
    valFolder = "val"
    normalFolder = "NORMAL"
    pneumoniaFolder = "PNEUMONIA"
    source = "local"

def getTrainData(source, numChannels):

    return getDataFrom(source, numChannels, RetrieverOpts.trainFolder)

def getTestData(source, numChannels):

    return getDataFrom(source, numChannels, RetrieverOpts.testFolder)

def getDataFrom(source, numChannels, dataType):

    if source == "gcp":
        return getDataFromGCP(dataType, numChannels)
    elif source == "local":
        return getDataFromLocalDisk(dataType, numChannels)
    else:
        mypprint.printError("Source " + source + " not known. Can not retrieve data")

    return None

def getDataFromGCP(dataType, numChannels):

    mypprint.printText("Retrieving data from Google Cloud Platform")

    return ("FOTO1", "1")

def getDataFromLocalDisk(dataType, numChannels):

    x = []
    y = []

    path = os.path.join(RetrieverOpts.dataFolder, dataType, RetrieverOpts.normalFolder)
    filesNames = os.listdir(path)

    for fileName in filesNames:
        if fileName.endswith(".jpeg"):
            if numChannels == 1:
                x.append(np.array(np.expand_dims(cv2.imread(os.path.join(path, fileName), cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imread(os.path.join(path, fileName))))
            y.append(0)

    path = os.path.join(RetrieverOpts.dataFolder, dataType, RetrieverOpts.pneumoniaFolder)
    filesNames = os.listdir(path)

    for fileName in filesNames:

        if fileName.endswith(".jpeg"):
            if numChannels == 1:
                x.append(np.array(np.expand_dims(cv2.imread(os.path.join(path, fileName), cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imread(os.path.join(path, fileName))))
            y.append(1)

    return (np.array(x), np.array(y))