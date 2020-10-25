import cv2
import numpy as np



class ProcessorOpts:
    normalize = True
    height = 1200
    width = 1200
    channels = 3


def processImages(images, opts):

    xSize = len(images)
    ySize = (opts.height, opts.width, opts.channels)

    dim = (xSize,) + ySize

    imagesProccesed = np.zeros(dim, float)

    for index, image in enumerate(images):
        
        imagesProccesed[index] = processImage(image, opts)
        images[index] = None

    return imagesProccesed


def processImage(image, opts):

    image = reshapeImage(image, opts)

    if opts.normalize == True:
        image = normalizeImage(image)
    
    return image


def parseRequestImage(image, opts):
    
    if opts.channels == 1:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRc)

    return processImage(image, opts)


def normalizeImage(image):

    return image / 255.


def reshapeImage(image, opts):

    if opts.channels == 3:
        return cv2.resize(image, (opts.height, opts.width))
    else:
        return np.expand_dims(cv2.resize(image, (opts.height, opts.width)), axis = 2)
