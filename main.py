# Main file of image recognition calculator (to be named later!)
# Created by: Brad Powell, Colin Lemarchand

import tensorflow as tf
import numpy as np
from PIL import Image
import os
from skimage.transform import resize

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
    #testLabelsFile = open("test-data/test-labels/labels.csv")
    #testLabelsString = testLabelsFile.read()
    #testImages = []
    #testLabels = testLabelsString.split(",")
    #for i in range(len(testLabels)):
    #    image = Image.open("test-data/test-images/test" + str(i) + ".png")
    #    testImages.append(convert_image_to_ndarray(image))
    #testImages = np.array(testImages)
    #testImages = tf.keras.utils.normalize(testImages, axis=1).reshape(testImages.shape[0], -1)

    #mlmodel = tf.keras.models.load_model('test_handwritten_num_reader2.model')
    #predictions = mlmodel.predict(testImages)
    #for i in range(len(predictions)):
    #    print(str(np.argmax(predictions[i])) + " ")
    
    
    data_path = 'test-data/'
    all_images = []
    all_labels = []
    for i in range(10):
        data_path_temp = 'test-data/' + str(i) + '/'
        dirlist = os.listdir(data_path_temp)
        for item in dirlist:
            image_file = Image.open(data_path_temp + item)
            image_ndarray = convert_image_to_ndarray(image_file)
            image_ndarray = resize(image_ndarray, (round(image_ndarray.shape[0] / 45 * 28), round(image_ndarray.shape[1] / 45 * 28)), anti_aliasing=True)
            all_images.append(image_ndarray)
            all_labels.append(i)
    all_images = np.array(all_images)
    all_images = tf.keras.utils.normalize(all_images, axis = 1).reshape(all_images.shape[0], -1)

    mlmodel = tf.keras.models.load_model('test_handwritten_num_reader.model')
    predictions = mlmodel.predict(all_images)
    successes = 0
    total = 0
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == all_labels[i]:
            successes += 1
        total += 1
    print("Accuracy: " + str(successes / total))
