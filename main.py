# Main file of image recognition calculator (to be named later!)
# Created by: Brad Powell, Colin Lemarchand

import tensorflow as tf
import numpy as np
from PIL import Image
import parse_input


if __name__ == "__main__":
    test_line_digits_image = Image.open("test.jpg")
    testing_images = parse_input.split_input_image(test_line_digits_image)
    temp_testing_images = []
    for image in testing_images:
        image = parse_input.format_image(image)
        temp_testing_images.append(image)
    testing_images = np.array(temp_testing_images)
    testing_images = tf.keras.utils.normalize(testing_images, axis=1).reshape(testing_images.shape[0], -1)

    mlmodel = tf.keras.models.load_model('handwritten_number_reader.model')
    predictions = mlmodel.predict(testing_images)

    for i in range(len(predictions)):
        print(np.argmax(predictions[i]))
