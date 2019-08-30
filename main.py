# Main file of image recognition calculator (to be named later!)i
# Created by: Brad Powell, Colin Lemarchand

#TODO:  Devise method to convert images into csv files in same format as MNIST training data

import tensorflow as tf
import numpy as np
from PIL import Image

def convert_image_to_ndarray(image):
    ''' Converts an image (28x28) to a NumPy ndarray object '''

    # Ensure that image is in grayscale
    image = image.convert('L')

    # Add all pixels to 2D list
    (w, h) = image.size
    imageList = [[0 for x in range(w)] for y in range(h)]
    for x in range(w):
        for y in range(h):
            imageList[y][x] = 255 - image.getpixel((x, y))

    imageNdArray = np.array(imageList)
    return imageNdArray


if __name__ == "__main__":
    #mnist = tf.keras.datasets.mnist
    #(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

    testLabelsFile = open("test-data/test-labels/labels.csv")
    testLabelsString = testLabelsFile.read()
    testImages = []
    testLabels = testLabelsString.split(",")
    for i in range(len(testLabels)):
        image = Image.open("test-data/test-images/test" + str(i) + ".png")
        testImages.append(convert_image_to_ndarray(image))
    testImages = np.array(testImages)
    testImages = tf.keras.utils.normalize(testImages, axis=1).reshape(testImages.shape[0], -1)

    mlmodel = tf.keras.models.load_model('test_handwritten_num_reader.model')
    predictions = mlmodel.predict(testImages)
    for i in range(len(predictions)):
        print(str(np.argmax(predictions[i])) + " ")
