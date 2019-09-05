# Main file of image recognition calculator (to be named later!)
# Created by: Brad Powell, Colin Lemarchand

import tensorflow as tf
import numpy as np
from PIL import Image
import parse_input


def get_predictions(predictions):
    predictions_string = ""
    for i in range(len(predictions)):
        prediction = np.argmax(predictions[i])
        if prediction < 10:
            predictions_string += str(prediction)
        elif prediction == 10:
            predictions_string += "+"
        elif prediction == 11:
            predictions_string += "-"
        elif prediction == 12:
            predictions_string += "*"
        elif prediction == 13:
            predictions_string += "/"
        else:
            predictions_string += "%"
    return predictions_string


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

    predicted_expression = get_predictions(predictions)
    try:
        print(predicted_expression + " = " + str(eval(predicted_expression)))
    except SyntaxError:
        print("Either you have entered an invalid input, or the model misunderstood your input.")
        print("The computer understood your input as: " + predicted_expression)
        print("If this is what you intended, please fix your input.")
        print("If the computer misunderstood your input, please take a new photo of your intended input.")
