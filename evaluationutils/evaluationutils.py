import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def saveModelInfo(processorOpts, trainningOpts, preprocessingTime, trainningTime, metrics, confusionMatrix, classes, folderPath):

    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    
    parametersFilePath = os.path.join(folderPath, "parametersSummary.json")
    cmFilePath = os.path.join(folderPath, "confusionMatrix.png")
    modelSummaryFilePath = os.path.join(folderPath, "modelSummary.txt")


    parameters = {
        "Training seconds" : trainningTime,
        "Processing seconds" : preprocessingTime,
        "Accuracy" : metrics[1]*100,
        "Loss" : metrics[0],
        "Image height" : processorOpts.height,
        "Image width" : processorOpts.width,
        "Epochs" : trainningOpts.epochs,
        "Batch Size" : trainningOpts.batchSize,
        "Channels" : processorOpts.channels,
        "Validation split " : trainningOpts.validationSplit,
        "Normalize" : processorOpts.normalize
    }

    with open(parametersFilePath, "w") as fp:
        json.dump(parameters, fp)
        
    saveConfusionMatrix(confusionMatrix, classes, cmFilePath)
    
def saveConfusionMatrix (cm, classes, filePath, normalize = False, title = "Confusion Matrix", cmap = plt.cm.Blues ):

    plt.clf()
    plt.imshow(cm, interpolation="nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()

    tickMarks = np.arange(len(classes))
    plt.xticks(tickMarks, classes, rotation = 45)
    plt.yticks(tickMarks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis = 1) [:, np.newaxis]

    thresh = cm.max() / 2.

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i,j],
            horizontalalignment = "center",
            color = "white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.savefig(filePath)

def saveModelHistory(historyTrain, folderPath):

    plt.clf()
    plt.plot(historyTrain.history["acc"])
    plt.plot(historyTrain.history["val_acc"])
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "test"], loc = "upper left")
    plt.savefig(os.path.join(folderPath, "Accuracy.png"))

    plt.clf()
    plt.plot(historyTrain.history["loss"])
    plt.plot(historyTrain.history["val_loss"])
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "test"], loc = "upper left")
    plt.savefig(os.path.join(folderPath, "Loss.png"))
    
