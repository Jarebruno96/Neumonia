from mypprint import mypprint
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

    # Reshape image
    image = reshapeImage(image, opts)

    # Normalize image
    if opts.normalize == True:
        image = normalizeImage(image)
    
    return image

def normalizeImage(image):
    return image / 255.

def reshapeImage(image, opts):
    if opts.channels == 3:
        return cv2.resize(image, (opts.height, opts.width))
    else:
        return np.expand_dims(cv2.resize(image, (opts.height, opts.width)), axis = 2)
