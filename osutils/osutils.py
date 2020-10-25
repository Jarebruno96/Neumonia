import os
import sys
import getopt
from mypprint import mypprint 
from chestxrayio.chestxrayio import IOOpts, GCP_SOURCE
from chestxrayprocessor.chestxrayprocessor import ProcessorOpts
from models.BinaryClassificationModel import TrainingOpts


def parseArgs(argv):

    ioOpts = IOOpts()
    processorOpts = ProcessorOpts()
    trainningOpts = TrainingOpts()

    try:
        opts, args = getopt.getopt(argv, 
                                    "hs:n:H:w:e:v:c:k:b:u:f:", 
                                    ["help", 
                                     "source=", 
                                     "normalize=", 
                                     "height=", 
                                     "width=", 
                                     "epochs=", 
                                     "validationSplit=",
                                     "channel=",
                                     "gcpkey=",
                                     "bucket=",
                                     "batch_size=",
                                     "folder="
                                     ]
                                    )
    except getopt.GetoptError as err:
        sys.exit(2)

    for opt, value in opts:

        if opt in ("-h", "--help"):
            mypprint.printText("Read README.md to know more about parameters")
            sys.exit()

        if opt in ("-s", "--source"):
            if value.lower() in ("local", "gcp"):
                ioOpts.source = value
            else:
                mypprint.printError("Source "+ value + " is not known")
                sys.exit(2)
        
        if opt in ("-n", "--normalize"):
            if value.lower() == "no":
                processorOpts.normalize = False
            elif value.lower() == "yes":
                processorOpts.normalize = True
            else:
                mypprint.printError("Normalize image`s pixels option "+ value + " is not known")
                sys.exit(2)
        
        if opt in ("-H", "--height"):
            try:
                processorOpts.height = int(value)
            except ValueError as err:
                mypprint.printError("Height argument not valid.")
                sys.exit(2)

        if opt in ("-w", "--width"):
            try:
                processorOpts.width = int(value)
            except ValueError as err:
                mypprint.printError("Width argument not valid.")
                sys.exit(2)
        
        if opt in ("-e", "--epochs"):
            try:
                trainningOpts.epochs = int(value)
            except ValueError as err:
                mypprint.printError("Epochs argument not valid.")
                sys.exit(2)
        
        if opt in ("-b", "--batch_size"):
            try:
                trainningOpts.batchSize = int(value)
            except ValueError as err:
                mypprint.printError("Batch Size argument not valid.")
                sys.exit(2)

        if opt in ("-v", "--validationSplit"):
            try:
                trainningOpts.validationSplit = float(value)
                if trainningOpts.validationSplit > 1 and trainningOpts.validationSplit < 0:
                    mypprint.printError("Validations split must be a value between 0 and 1")
                    sys.exit(2)
            except ValueError as err:
                mypprint.printError("Validation split argument not valid.")
                sys.exit(2)
            
        if opt in ("-c", "--channel"):
            if value.lower() == "rgb":
                processorOpts.channels = 3
            elif value.lower() == "gray":
                processorOpts.channels = 1
            else:
                sys.exit(2)
        
        if opt in ("-k", "--gcpkey"):
            ioOpts.googleStorageKey = value
        
        if opt in ("-u", "--bucket"):
            ioOpts.bucketName = value
        
        if opt in ("-f", "--folder"):

            if value[-1] == os.sep:
                value = value[:-1]
            ioOpts.modelFolder = value

    return ioOpts, processorOpts, trainningOpts
    

def printArguments(ioOpts, processorOpts, trainningOpts):

    mypprint.printBlue("######################################################################")
    mypprint.printBlue("## Reading images from " + ioOpts.source + " source")
    if ioOpts.source == GCP_SOURCE:
        mypprint.printBlue("## Bucket name: " + ioOpts.bucketName)
    mypprint.printBlue("## Images options:")
    mypprint.printBlue("##     Dimesions: " + str(processorOpts.height)  + "x" + str(processorOpts.width))
    mypprint.printBlue("##     Channels: " + str(processorOpts.channels) + " " + ("(RGB)" if processorOpts.channels == 3 else "(Gray)"))
    mypprint.printBlue("##     Pixels normalize: " + str(processorOpts.normalize))
    mypprint.printBlue("## Trainning options:")
    mypprint.printBlue("##     Epochs: " + str(trainningOpts.epochs))
    mypprint.printBlue("##     Batch Size: " + str(trainningOpts.batchSize))
    mypprint.printBlue("##     Validation Split %: " + str(trainningOpts.validationSplit))
    if ioOpts.source == GCP_SOURCE:
        mypprint.printBlue("## Folder to store model: " + ioOpts.bucketName + "/" + ioOpts.modelFolder)
        mypprint.printBlue("## Google Key file: " + (ioOpts.googleStorageKey if ioOpts.googleStorageKey else "Implicit permission"))
    else:
        mypprint.printBlue("## Folder to store model: " + (ioOpts.modelFolder if ioOpts.modelFolder else "."))
    mypprint.printBlue("######################################################################")


def createFolder(folderPath):

    if not os.path.isdir(folderPath):
        try:
            os.mkdir(folderPath)
        except OSError as err:
            mypprint.printError("Can not create directory " % folderPath)
            sys.exit(2)
