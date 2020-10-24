import os
from mypprint import mypprint 
import cv2
import numpy as np
from datetime import datetime
from google.cloud import storage
import json

GCP_SOURCE = "gcp"
LOCAL_SOURCE = "local"
GRAY_SCALE_CHANNELS = 1
JPEG_CONTENT_TYPE = "image/jpeg"

class IOOpts:
    dataFolder = "chest_xray"
    trainFolder = "train"
    testFolder = "test"
    valFolder = "val"
    normalFolder = "NORMAL"
    pneumoniaFolder = "PNEUMONIA"
    source = "local"
    googleStorageKey = ""
    bucketName = ""
    modelFolder = datetime.today().strftime("%d-%m-%y") + "_saves"
    bucket = None


def getTrainData(ioOpts, processorOpts):

    return getDataFrom(ioOpts, processorOpts, ioOpts.trainFolder)


def getTestData(ioOpts, processorOpts):

    return getDataFrom(ioOpts, processorOpts, ioOpts.testFolder)


def getDataFrom(ioOpts, processorOpts, dataType):

    if ioOpts.source == GCP_SOURCE:
        return getDataFromGCP(dataType, processorOpts, ioOpts)
    elif ioOpts.source == LOCAL_SOURCE:
        return getDataFromLocalDisk(dataType, processorOpts, ioOpts)
    else:
        mypprint.printError("Source " + source + " not known. Can not retrieve data")

    return (np.array([]), np.array([]))


def getGCPStorageBucket(ioOpts):

    if not ioOpts.googleStorageKey:
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(ioOpts.googleStorageKey)
    
    return client.bucket(ioOpts.bucketName)


def getDataFromGCP(dataType, processorOpts, ioOpts):

    x = []
    y = []

    blobs = ioOpts.bucket.list_blobs(prefix = ioOpts.dataFolder + "/" + dataType + "/" + ioOpts.normalFolder)

    for blob in blobs:
        if blob.content_type == JPEG_CONTENT_TYPE:

            if processorOpts.channels == GRAY_SCALE_CHANNELS:
                image = np.asarray(bytearray(blob.download_as_string()), dtype="uint8")
                x.append(np.array(np.expand_dims(cv2.imdecode(image, cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imdecode(image)))
            y.append(0)
    

    blobs = ioOpts.bucket.list_blobs(prefix = ioOpts.dataFolder + "/" + dataType + "/" + ioOpts.pneumoniaFolder)

    for blob in blobs:
        if blob.content_type == JPEG_CONTENT_TYPE:

            if processorOpts.channels == GRAY_SCALE_CHANNELS:
                image = np.asarray(bytearray(blob.download_as_string()), dtype="uint8")
                x.append(np.array(np.expand_dims(cv2.imdecode(image, cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imdecode(image)))
            y.append(1)
    
    return (np.array(x), np.array(y))


def getDataFromLocalDisk(dataType, processorOpts, ioOpts):

    x = []
    y = []

    path = os.path.join(ioOpts.dataFolder, dataType, ioOpts.normalFolder)
    filesNames = os.listdir(path)

    for fileName in filesNames:
        if fileName.endswith(".jpeg"):
            if processorOpts.channels == GRAY_SCALE_CHANNELS:
                x.append(np.array(np.expand_dims(cv2.imread(os.path.join(path, fileName), cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imread(os.path.join(path, fileName))))
            y.append(0)

    path = os.path.join(ioOpts.dataFolder, dataType, ioOpts.pneumoniaFolder)
    filesNames = os.listdir(path)

    for fileName in filesNames:

        if fileName.endswith(".jpeg"):
            if processorOpts.channels == GRAY_SCALE_CHANNELS:
                x.append(np.array(np.expand_dims(cv2.imread(os.path.join(path, fileName), cv2.IMREAD_GRAYSCALE), axis = 2)))
            else:
                x.append(np.array(cv2.imread(os.path.join(path, fileName))))
            y.append(1)

    return (np.array(x), np.array(y))


def uploadFileTOGCPStorage(fileName, ioOpts):

    blob = ioOpts.bucket.blob(ioOpts.modelFolder + os.sep + fileName)
    blob.upload_from_filename(fileName)


def saveImage(imageName, image, ioOpts):
    
    if ioOpts.source == LOCAL_SOURCE:
        cv2.imwrite(os.path.join(ioOpts.modelFolder, imageName), image)
    else:
        cv2.imwrite(imageName, image)
        uploadFileTOGCPStorage(imageName, ioOpts)
        deleteFile(imageName)


def saveFigure(figureName, figure, ioOpts):
    
    if ioOpts.source == LOCAL_SOURCE:
        figure.savefig(os.path.join(ioOpts.modelFolder, figureName))
    else:
        figure.savefig(figureName)
        uploadFileTOGCPStorage(figureName, ioOpts)
        deleteFile(figureName)
    
    #figure.clf()


def saveMetrics(fileName, trainingOpts, processorOpts, timeMetrics, modelMetrics, ioOpts):

    metrics = {
        "Model metrics" : modelMetrics,
        "Time Metrics" : timeMetrics,
        "Training Options" : {
            "Validation split " : trainingOpts.validationSplit,
            "Epochs" : trainingOpts.epochs,
            "Batch Size" : trainingOpts.batchSize
        },
        "Preproccesing Options": {
            "Image height" : processorOpts.height,
            "Image width" : processorOpts.width,
            "Normalize" : processorOpts.normalize,
            "Channels" : processorOpts.channels
        }
    }

    if ioOpts.source == LOCAL_SOURCE:
        with open(os.path.join(ioOpts.modelFolder, fileName), "w") as fp:
            json.dump(metrics, fp)
    else:
        with open(fileName, "w") as fp:
            json.dump(metrics, fp)
        uploadFileTOGCPStorage(fileName, ioOpts)
        deleteFile(fileName)


def saveModel(fileName, model, ioOpts):

    if ioOpts.source == LOCAL_SOURCE:       
        model.save(os.path.join(ioOpts.modelFolder, fileName))
    else:
        model.save(fileName)
        uploadFileTOGCPStorage(fileName, ioOpts)
        deleteFile(fileName)


def saveModelSummary(fileName, model, ioOpts):
    
    summary = model.getModelSummary()

    if ioOpts.source == LOCAL_SOURCE:
        with open(os.path.join(ioOpts.modelFolder, fileName), "w") as fp:
            fp.write(summary)
    else:
        with open(fileName, "w") as fp:
            fp.write(summary)
        uploadFileTOGCPStorage(fileName, ioOpts)
        deleteFile(fileName)


def deleteFile(fileName):

    try:
        os.remove(fileName)
    except OSError as e:
        mypprint.printError("Can not delete " + fileName + ". " + str(e))