# Main file of image recognition calculator (to be named later!)
# Created by: Brad Powell, Colin Lemarchand

import tensorflow as tf
import numpy as np
from PIL import Image
import PIL.ImageOps
import os
from skimage.transform import resize
import parse_input


def load_image(path):
    ''' Loads an image from a given path (containing the image's filename) and formats it for input into the model '''

    # Open the image
    image = Image.open(path)
    
    # Convert the image to grayscale if it wasn't already (TODO: Change to black & white)
    image = image.convert('L')
    
    # Invert the image
    inverted_image = PIL.ImageOps.invert(image)

    # Convert the inverted image into an ndarray
    image_ndarray = np.array(inverted_image)

    # Resize the image to have the same resolution as the training data
    image_ndarray = resize(image_ndarray, (20, 20), anti_aliasing=True)

    # Add four pixels of padding to all sides of the image 
    image_list_temp = [[0 for x in range(28)] for y in range(28)]
    for x in range(image_ndarray.shape[0]):
        for y in range(image_ndarray.shape[1]):
            image_list_temp[4 + y][4 + x] = image_ndarray[y][x]
    image_ndarray = np.array(image_list_temp)

    return image_ndarray


if __name__ == "__main__":
    
    data_path = 'training-data/hasy-data/v2-00'

    testing_images = []
    testing_labels = []

    for i in range(104):
            item_data_path = data_path + str(i+345) + ".png"
            image = load_image(item_data_path)
            testing_images.append(image)
            #testing_labels.append(i)
    testing_images = np.array(testing_images)
    testing_images = tf.keras.utils.normalize(testing_images, axis = 1).reshape(testing_images.shape[0], -1)

    mlmodel = tf.keras.models.load_model('handwritten_number_reader.model')
    predictions = mlmodel.predict(testing_images)

    labels_path = 'training-data/hasy-labels/test-labels.csv'
    labels = open(labels_path)
    labels_string = labels.read()
    labels_string = labels_string.split()

    for i in range(len(predictions)):
        print(np.argmax(predictions[i]))
    successes = 0
    total = 0
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == int(labels_string[i]):
            successes += 1
        total += 1
    print("Accuracy: " + str(successes / total))
