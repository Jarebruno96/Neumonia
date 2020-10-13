import sys
import getopt
from mypprint import mypprint 
from models.LocalModel import LocalModel, TrainningOpts
from chestxrayretriever import chestxrayretriever
import numpy as np
from chestxrayprocessor import chestxrayprocessor
from chestxrayprocessor.chestxrayprocessor import ProcessorOpts
import time
from datetime import datetime
from keras import backend as K
from tensorflow.python.client import device_lib
from evaluationutils import evaluationutils
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator

def usage():
    mypprint.printText("main.py [-s data_source]")

def main(argv):

    mypprint.printHeader("Neumonia use case")

    # Checking arguments

    retrieverOpts = chestxrayretriever.RetrieverOpts()
    processorOpts = ProcessorOpts()
    trainningOpts = TrainningOpts()

    # Reading and parsing args
    try:
        opts, args = getopt.getopt(argv, 
                                    "hs:n:H:w:e:v:f:c:", 
                                    ["help", 
                                     "source=", 
                                     "normalize=", 
                                     "height=", 
                                     "width=", 
                                     "epochs=", 
                                     "validationSplit=",
                                     "folder=",
                                     "channel="]
                                    )
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)

    for opt, value in opts:

        if opt in ("-h", "--help"):
            usage()
            sys.exit()

        if opt in ("-s", "--source"):
            if value.lower() in ("local", "gcp"):
                retrieverOpts.source = value
            else:
                mypprint.printError("Source "+ value + " is not known")
                usage()
                sys.exit(2)
        
        if opt in ("-n", "--normalize"):
            if value.lower() == "no":
                processorOpts.normalize = False
            elif value.lower() == "yes":
                processorOpts.normalize = True
            else:
                mypprint.printError("Normalize image`s pixels option "+ value + " is not known")
                usage()
                sys.exit(2)
        
        if opt in ("-H", "--height"):
            try:
                processorOpts.height = int(value)
            except ValueError as err:
                usage()
                mypprint.printError("Height argument not valid.")
                sys.exit(2)

        if opt in ("-w", "--width"):
            try:
                processorOpts.width = int(value)
            except ValueError as err:
                usage()
                mypprint.printError("Width argument not valid.")
                sys.exit(2)
        
        if opt in ("-e", "--epochs"):
            try:
                trainningOpts.epochs = int(value)
            except ValueError as err:
                usage()
                mypprint.printError("Epochs argument not valid.")
                sys.exit(2)
        
        if opt in ("-b", "--batch_size"):
            try:
                trainningOpts.batchSize = int(value)
            except ValueError as err:
                usage()
                mypprint.printError("Batch Size argument not valid.")
                sys.exit(2)

        if opt in ("-v", "--validationSplit"):
            try:
                trainningOpts.validationSplit = float(value)
                if trainningOpts.validationSplit > 1 and trainningOpts.validationSplit < 0:
                    mypprint.printError("Validations split must be a value between 0 and 1")
                    usage()
                    sys.exit(2)
            except ValueError as err:
                usage()
                mypprint.printError("Validation split argument not valid.")
                sys.exit(2)

        if opt in ("-f", "--folder"):
            saveFolder = value
            
        if opt in ("-c", "--channel"):
            if value.lower() == "rgb":
                processorOpts.channels = 3
            elif value.lower() == "gray":
                processorOpts.channels = 1
            else:
                mypprint.printError("Channels option"+ value + " is not known")
                usage()
                sys.exit(2)


    mypprint.printBlue("######################################################################")
    mypprint.printBlue("## Reading images from " + retrieverOpts.source + " source")
    mypprint.printBlue("## Images options:")
    mypprint.printBlue("##     Dimesions: " + str(processorOpts.height)  + "x" + str(processorOpts.width))
    mypprint.printBlue("##     Channels: " + str(processorOpts.channels) + " " + ("(RGB)" if processorOpts.channels == 3 else "(Gray)"))
    mypprint.printBlue("##     Pixels normalize: " + str(processorOpts.normalize))
    mypprint.printBlue("## Trainning options:")
    mypprint.printBlue("##     Epochs: " + str(trainningOpts.epochs))
    mypprint.printBlue("##     Batch Size: " + str(trainningOpts.batchSize))
    mypprint.printBlue("##     Validation Split %: " + str(trainningOpts.validationSplit))
    mypprint.printBlue("######################################################################")

    detectPneumonia(retrieverOpts, processorOpts, trainningOpts)


def detectPneumonia(retrieverOpts, processorOpts, trainningOpts):

    # Create folder to save model info
    saveFolder = datetime.today().strftime("%d-%m-%y") + "_saves"
    if not os.path.isdir(saveFolder):
        try:
            os.mkdir(saveFolder)
        except OSError as err:
            mypprint.printError("Can not create directory " % saveFolder)
            sys.exit(2)
    
    #Getting data for training

    mypprint.printText("Reading images files from " + retrieverOpts.source + " source")
    xTrain, yTrain = chestxrayretriever.getTrainData(retrieverOpts.source, processorOpts.channels)
    xTest, yTest = chestxrayretriever.getTestData(retrieverOpts.source, processorOpts.channels)

    mypprint.printText("Number of normal samples: " + str(np.count_nonzero(yTrain==0)))
    mypprint.printText("Number of pneumonia samples: " + str(np.count_nonzero(yTrain)))

    # Save original sample
    cv2.imwrite(os.path.join(saveFolder, "originalSample.jpeg"),xTrain[0])
    
    # Processing images

    startProcessing = time.time()

    mypprint.printText("Preproccesing train images")
    xTrain = chestxrayprocessor.processImages(xTrain, processorOpts)

    mypprint.printText("Preproccesing test images")
    xTest = chestxrayprocessor.processImages(xTest, processorOpts)

    endProcessing = time.time()
    preprocessingTime = round(endProcessing-startProcessing,2)

    mypprint.printBlue(str(len(yTest) + len(yTrain)) + " samples processed in " + str(preprocessingTime))

    # As there are less images from normal, we create new ones from the originals (1 -> 2)

    datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = "nearest"
    )

    datagen.fit(xTrain)

    # Save original sample processed
    cv2.imwrite(os.path.join(saveFolder, "processedSample.jpeg"), xTrain[0])

    # Creating model

    model = LocalModel((processorOpts.height, processorOpts.width, processorOpts.channels))

    # Training model

    mypprint.printBlue("Trainning model")
    startTrain = time.time()
    historyTrain = model.train(xTrain, yTrain, datagen, trainningOpts)
    endTrain = time.time()
    mypprint.printBlue("Saving model")
    model.save(os.path.join(saveFolder, "model.h5"))
    evaluationutils.saveModelHistory(historyTrain, saveFolder)

    # Predicting with model

    mypprint.printBlue("Making predictions on " + str(len(xTest)) + " samples")
    yTestPredicted = model.predict(xTest)

    # Saving conclusions model
    preprocessingTime = round(endProcessing-startProcessing,2)
    trainningTime = round(endTrain-startTrain,2)

    for index, prediction in enumerate(yTestPredicted):
        
        if prediction <= 0.5:
            yTestPredicted[index] = 0
        else:
            yTestPredicted[index] = 1

    classes = ["Normal", "Pneumonia"]
    savesFolder = "saves"

    confusionMatrix, metrics = model.evaluate(xTest, yTest, yTestPredicted, datagen, trainningOpts)

    modelSummary = model.saveModelSummary(saveFolder)
    evaluationutils.saveModelInfo(processorOpts, trainningOpts, preprocessingTime, trainningTime, metrics, confusionMatrix, classes, saveFolder)


if __name__ == "__main__":
    main(sys.argv[1:])