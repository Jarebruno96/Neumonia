from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from datetime import date
import os
from mypprint import mypprint
from sklearn.model_selection import train_test_split

class TrainningOpts:
    epochs = 10
    batchSize = 32
    validationSplit = 0.2


class LocalModel():

    def __init__(self, inputShape):

        self.model = Sequential()

        self.model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding = "same",  input_shape=inputShape))
        self.model.add(MaxPooling2D(pool_size = (2,2), padding = "same"))

        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding = "same"))
        self.model.add(MaxPooling2D(pool_size = (2,2), padding = "same"))

        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding = "same"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size = (2,2), padding = "same"))

        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding = "same"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size = (2,2), padding = "same"))

        self.model.add(Flatten())

        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()


    def train(self, xTrain, yTrain, datagenerator, opts = TrainningOpts()):

        x_train, x_val, y_train, y_val = train_test_split(xTrain, yTrain, test_size = opts.validationSplit)

        return self.model.fit_generator(
            datagenerator.flow(x_train, y_train, batch_size = opts.batchSize),
            validation_data =  (x_val, y_val),
            epochs = opts.epochs, 
            steps_per_epoch = len(yTrain)/opts.batchSize)

    def predict(self, xVal):
        return self.model.predict(xVal)

    def evaluate(self, xTest, yTest, yTestPredicted, datagenerator, opts = TrainningOpts()):
        matrix = confusion_matrix(yTest, yTestPredicted)
        metrics = self.model.evaluate_generator(datagenerator.flow(xTest, yTest), steps = len(yTest)/opts.batchSize)
        return matrix, metrics

    def loadFromFile(self, file):
        try:
            self.model = load_model(file)
        except OSError as err:
            mypprint.printError("Can not load model from " + file)
            return False
        return True

    def save(self, modelFile):        
        self.model.save(modelFile)
    
    def saveModelSummary(self, folder):
        with open(os.path.join(folder, "summary.txt"), "w") as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + '\n'))

