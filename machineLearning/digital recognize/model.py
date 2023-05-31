
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import linear, relu, sigmoid
import warnings

from read_file import read_idx3_file

warnings.simplefilter(action='ignore', category=FutureWarning)

images_file = 'train-images.idx3-ubyte'
labels_file = 'train-labels.idx1-ubyte'

images_data, labels_data = read_idx3_file(images_file, labels_file)

tf.random.set_seed(1235)
model = Sequential(
    [
        tf.keras.Input(shape=(784,)),
        Dense(25, activation="relu"),
        Dense(15, activation="relu"),
        Dense(10, activation="linear"),
    ], name="my_model"
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
model.fit(images_data, labels_data, epochs=70)
model.save('model.h5')
