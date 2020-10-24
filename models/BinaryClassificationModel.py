from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



class TrainingOpts:
    epochs = 10
    batchSize = 32
    validationSplit = 0.2


class BinaryClassificationModel():

    def __init__(self, processorOpts):

        inputShape = (processorOpts.height, processorOpts.width, processorOpts.channels)

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


    def train(self, xTrain, yTrain, datagenerator, opts = TrainingOpts()):

        x_train, x_val, y_train, y_val = train_test_split(xTrain, yTrain, test_size = opts.validationSplit)

        return self.model.fit_generator(
            datagenerator.flow(x_train, y_train, batch_size = opts.batchSize),
            validation_data =  (x_val, y_val),
            epochs = opts.epochs, 
            steps_per_epoch = len(yTrain)/opts.batchSize)

    def save(self, path):
    
        self.model.save(path)
    
    def getModelSummary(self):
        
        lines = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        
        return "\n".join(lines)


    def predict(self, xVal):

        return self.model.predict(xVal)

    def parsePredicctions(self, predictions):

        for index, prediction in enumerate(predictions):
            
            if prediction <= 0.5:
                predictions[index] = 0
            else:
                predictions[index] = 1
        
        return predictions


    def getMetrics(self, cases, solutions, datagenerator, opts = TrainingOpts()):

        return self.model.evaluate_generator(datagenerator.flow(cases, solutions), steps = len(solutions)/opts.batchSize)

    
    def getConfusionMatrix(self, solutions, predictions):
        
        return confusion_matrix(solutions, predictions)