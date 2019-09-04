import tensorflow as tf
import numpy as np
from PIL import Image
from main import load_image
import os
from skimage.transform import resize

mnist = tf.keras.datasets.mnist
(training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()

extra_data = []
extra_labels = []


def load_from_sorted_dir(label, dirname):
    item_path = "training-data/" + str(dirname) + "/"
    dirlist = os.listdir(item_path)
    for item in dirlist:
        image = load_image(Image.open(item_path + item))
        extra_data.append(image)
        extra_labels.append(label)

for i in range(15):
    if i < 10:
        load_from_sorted_dir(str(i), i)
    elif i == 10:
        load_from_sorted_dir(10, "data-plus")
    elif i == 11:
        load_from_sorted_dir(11, "data-minus")
    elif i == 12:
        load_from_sorted_dir(12, "data-times")
    elif i == 13:
        load_from_sorted_dir(13, "data-divide")
    else:
        load_from_sorted_dir(14, "data-modulo")
training_data = np.append(training_data, np.array(extra_data), axis=0)
training_labels = np.append(training_labels, extra_labels)

# Open the semeion data file and prepare lists for input
even_more_data = open('training-data/semeion.data')
data_string = even_more_data.read()
data_string = data_string.split()
extra_data = []
temp_image = []
extra_labels = []

# Process the data in the semeion data file
index_in_line = 0
for num in data_string:
    if index_in_line < 256:  # First 256 characters on line are pixels
        temp_image.append(num)
        index_in_line += 1
    elif index_in_line == 256:  # Loading pixels complete; add image to queue
        extra_data.append(temp_image)
        temp_image = []
        index_in_line += 1
        if int(float(num)) == 1:  # If the first marker is 1, then the label is 0
            extra_labels.append(0)
    elif index_in_line < 265:  # Characters from indexes 256-265 are reserved for the image's label
        if int(float(num)) == 1:  # If the current marker is 1, then the label is the index (with respect to markers)
            extra_labels.append(index_in_line - 256)
        index_in_line += 1
    else:
        if int(float(num)) == 1:  # If the final marker is 1, then the label is 9
            extra_labels.append(9)
        index_in_line = 0
new_extra_data = []
for image in extra_data:
    x = 0
    y = 0
    image_list = []
    temp_list = []
    for pixel in image:
        if x == 16:
            x = 0
            y += 1
            image_list.append(np.array(temp_list))
            temp_list = []
        temp_list.append(255 * int(float(pixel)))
        x += 1
    image_list.append(np.array(temp_list))
    image_list = np.array(image_list)
    image_list = resize(image_list, (20, 20), anti_aliasing=True)
    # add padding
    new_image_list = [[0 for x in range(28)] for y in range(28)]
    for x in range(20):
        for y in range(20):
            new_image_list[4 + y][4 + x] = image_list[y][x]
    new_image_list = np.array(new_image_list)
    new_extra_data.append(new_image_list)
    image_list = []
new_extra_data = np.array(new_extra_data)
training_data = np.append(training_data, new_extra_data, axis=0)
training_labels = np.append(training_labels, extra_labels)

training_data = tf.keras.utils.normalize(training_data, axis=1).reshape(training_data.shape[0], -1)
testing_data = tf.keras.utils.normalize(testing_data, axis=1).reshape(testing_data.shape[0], -1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=training_data.shape[1:]))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(15, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=3)

val_loss, val_acc = model.evaluate(testing_data, testing_labels)
print("Loss: " + str(val_loss))
print("Accuracy: " + str(val_acc))

model.save("handwritten_number_reader.model")

print("Model is generated.")
