import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from main import load_image
import os

mnist = tf.keras.datasets.mnist
(training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()

extra_data = []
extra_labels = []

data_path = 'test-data/'
for i in range(10):
    item_path = data_path + str(i) + '/'
    dirlist = os.listdir(item_path)
    for item in dirlist:
        image = load_image(item_path + item)
        extra_data.append(image)
        extra_labels.append(i)
training_data = np.append(training_data, np.array(extra_data), axis = 0)
training_labels = np.append(training_labels, extra_labels)

training_data = tf.keras.utils.normalize(training_data, axis=1).reshape(training_data.shape[0], -1)
testing_data = tf.keras.utils.normalize(testing_data, axis=1).reshape(testing_data.shape[0], -1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu, input_shape = training_data.shape[1:]))
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'nadam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(training_data, training_labels, epochs = 3)

val_loss, val_acc = model.evaluate(testing_data, testing_labels)
print("Loss: " + str(val_loss))
print("Accuracy: " + str(val_acc))

model.save("handwritten_number_reader.model")

print("Model is generated.")
