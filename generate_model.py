import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()

training_data = tf.keras.utils.normalize(training_data, axis=1).reshape(training_data.shape[0], -1)
testing_data = tf.keras.utils.normalize(testing_data, axis=1).reshape(testing_data.shape[0], -1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu, input_shape = training_data.shape[1:]))
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(training_data, training_labels, epochs = 3)

val_loss, val_acc = model.evaluate(testing_data, testing_labels)
print("Loss: " + str(val_loss))
print("Accuracy: " + str(val_acc))

model.save("handwritten_number_reader.model")

print("Model is generated.")
