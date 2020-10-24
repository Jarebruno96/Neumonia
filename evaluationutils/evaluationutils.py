import numpy as np
import itertools
import matplotlib.pyplot as plt



def buildConfusionMatrixPlot (confusionMatrix, classes, xLabel = "", yLabel = "", title = "", normalize = False, colorMap = plt.cm.Blues ):

    #plt.clf()
    fig = plt.figure()

    plt.imshow(confusionMatrix, interpolation="nearest", cmap = colorMap)
    plt.title(title)
    plt.colorbar()

    tickMarks = np.arange(len(classes))
    plt.xticks(tickMarks, classes, rotation = 45)
    plt.yticks(tickMarks, classes)

    if normalize:
        confusionMatrix = confusionMatrix.astype("float") / confusionMatrix.sum(axis = 1) [:, np.newaxis]

    thresh = confusionMatrix.max() / 2.

    for i,j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):

        plt.text(j, i, confusionMatrix[i,j],
            horizontalalignment = "center",
            color = "white" if confusionMatrix[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(xLabel)
    plt.xlabel(yLabel)

    return fig


def buildPlot(title = "", xLabel = "", yLabel = "", legend = [""], series=[]):

    fig = plt.figure()

    for serie in series:
        plt.plot(serie)

    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(legend, loc = "upper left")

    return fig
