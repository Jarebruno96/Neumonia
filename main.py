import sys
import time
import numpy as np
from osutils import osutils
from mypprint import mypprint
from chestxrayio import chestxrayio
from models.BinaryClassificationModel import BinaryClassificationModel
from chestxrayprocessor import chestxrayprocessor
from keras.preprocessing.image import ImageDataGenerator
from evaluationutils import evaluationutils


def main(argv):

    # Reading, parsing and printing arguments
    
    ioOpts, processorOpts, trainingOpts = osutils.parseArgs(argv)
    osutils.printArguments(ioOpts, processorOpts, trainingOpts)

    # Create local folder or get bucket pointer

    if ioOpts.source == chestxrayio.LOCAL_SOURCE:
        osutils.createFolder(ioOpts.modelFolder)
    else:
        ioOpts.bucket = chestxrayio.getGCPStorageBucket(ioOpts)

    
    #Getting data for training

    mypprint.printText("Reading images files from " + ioOpts.source + " source")

    xTrain, yTrain = chestxrayio.getTrainData(ioOpts, processorOpts)
    xTest, yTest = chestxrayio.getTestData(ioOpts, processorOpts)

    mypprint.printText("Number of normal samples: " + str(np.count_nonzero(yTrain==0)))
    mypprint.printText("Number of pneumonia samples: " + str(np.count_nonzero(yTrain)))


    # Save original sample

    chestxrayio.saveImage("originalSample.jpeg", xTrain[0], ioOpts)

    
    # Preprocessing images

    mypprint.printText("Preproccesing images")

    startPreprocessingTime = time.time()

    xTrain = chestxrayprocessor.processImages(xTrain, processorOpts)
    xTest = chestxrayprocessor.processImages(xTest, processorOpts)

    endPreprocessingTime = time.time()
    
    elapsedPreprocessingTime = round(endPreprocessingTime-startPreprocessingTime,2)

    mypprint.printText(str(len(yTest) + len(yTrain)) + " samples processed in " + str(elapsedPreprocessingTime))


    # Generating more images from read images

    mypprint.printText("Generating more images from read images")

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


    # Save original sample preprocessed

    chestxrayio.saveImage("originalSamplePreprocesed.jpeg", xTrain[0], ioOpts)


    # Creating model

    model = BinaryClassificationModel(processorOpts)


    # Training model

    mypprint.printText("Training model")

    trainStartTime = time.time()
    historyTrain = model.train(xTrain, yTrain, datagen, trainingOpts)
    trainEndTine = time.time()

    elapsedTrainingTime = round(trainEndTine-trainStartTime,2)


    # Saving model

    mypprint.printText("Saving model")

    chestxrayio.saveModel("model.h5", model, ioOpts)
    chestxrayio.saveModelSummary("ModelSummary.txt", model, ioOpts)

    accuracyFigure = evaluationutils.buildPlot(title = "Model Accuracy", 
                                             xLabel = "Epoch", 
                                             yLabel = "Accuracy", 
                                             legend = ["train", "test"],
                                             series = [historyTrain.history["acc"], historyTrain.history["val_acc"]]
                                            )
    
    lossFigure = evaluationutils.buildPlot(title = "Model Loss", 
                                            xLabel = "Epoch", 
                                            yLabel = "Loss", 
                                            legend = ["train", "test"],
                                            series = [historyTrain.history["loss"], historyTrain.history["val_loss"]]
                                        )   
    
    chestxrayio.saveFigure("Accuracy.png", accuracyFigure, ioOpts)
    chestxrayio.saveFigure("Loss.png", lossFigure, ioOpts)


    # Predictions with model 
    
    mypprint.printText("Making predictions on " + str(len(xTest)) + " samples")
    yTestPredicted = model.predict(xTest)
    yTestPredictedParsed = model.parsePredicctions(yTestPredicted)


    # Confusion matrix

    confusionMatrix = model.getConfusionMatrix(yTest, yTestPredictedParsed)
    metrics = model.getMetrics(xTest, yTest, datagen, trainingOpts)

    confusionMatrixFigure = evaluationutils.buildConfusionMatrixPlot(confusionMatrix, 
                                                                   classes = ["Normal", "Pneumonia"],
                                                                   title = "Confusion Matrix",
                                                                   xLabel = "True label",
                                                                   yLabel = "Predicted label")
    
    chestxrayio.saveFigure("Confusion Matrix.png", confusionMatrixFigure, ioOpts)


    # Other metrics

    timeMetrics = {
        "Training seconds time" : elapsedTrainingTime,
        "Preproccesing seconds time" : elapsedPreprocessingTime
    }

    modelMetrics = {
        "Accuracy" : round(metrics[1]*100, 4),
        "Loss" : round(metrics[0], 4)
    }


    chestxrayio.saveMetrics("Model Parameters.json", trainingOpts, processorOpts, timeMetrics, modelMetrics, ioOpts)

if __name__ == "__main__":
    main(sys.argv[1:])