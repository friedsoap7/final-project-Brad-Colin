# Main file of image recognition calculator (to be named later!)
# Created by: Brad Powell, Colin Lemarchand

import tensorflow as tf
import numpy as np
from PIL import Image
import PIL.ImageOps
import os
from skimage.transform import resize
from parse_input import process_input_image


def load_image(image):
    ''' Formats an image for input into the model '''
    
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
    test_line_digits_image = Image.open("test.jpg")
    
    test_line_digits_image = test_line_digits_image.convert('L')
    for x in range(test_line_digits_image.size[0]):
        for y in range(test_line_digits_image.size[1]):
            if test_line_digits_image.getpixel((x, y)) < 80:
                test_line_digits_image.putpixel((x, y), 0)
            else:
                test_line_digits_image.putpixel((x, y), 255)
    test_line_digits_image.show()

    testing_images = process_input_image(test_line_digits_image)
    temp_testing_images = []
    for image in testing_images:
        image = load_image(image)
        temp_testing_images.append(image)
    testing_images = np.array(temp_testing_images)
    testing_images = tf.keras.utils.normalize(testing_images, axis = 1).reshape(testing_images.shape[0], -1)

    mlmodel = tf.keras.models.load_model('handwritten_number_reader.model')
    predictions = mlmodel.predict(testing_images)

    for i in range(len(predictions)):
        print(np.argmax(predictions[i]))
