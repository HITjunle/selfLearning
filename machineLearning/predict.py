import math

from tensorflow import keras
from keras.optimizers import Adam
import numpy as np
from read_file import read_idx3_file

model = keras.models.load_model('model.h5', custom_objects={'CustomAdam': Adam})
images_predict_file = 't10k-images.idx3-ubyte'
labels_predict_file = 't10k-labels.idx1-ubyte'
images_predict_data, labels_predict_data = read_idx3_file(images_predict_file, labels_predict_file)
prediction = model.predict(images_predict_data[0].reshape(1, 784))
print(f"the shape of input data is {images_predict_data.shape}")
print("Try to predict the first data")
print(f"the prediction is {np.argmax(prediction)}")
print(f"the true thing is {labels_predict_data[0]}")
print("Then I'd like to know the accuracy of the model, wait a moment:")

# 批处理预测
batch_size = 32  # 每个批次的大小
num_samples = len(images_predict_data)
num_batches = (num_samples + batch_size - 1) // batch_size  # 计算总批次数
num_sample = len(images_predict_data)
correct_prediction = 0
# 批处理
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_samples)
    batch_images = images_predict_data[start_idx:end_idx]
    batch_labels = labels_predict_data[start_idx:end_idx]

    # 预测当前批次
    batch_predictions = model.predict(batch_images, verbose=0)
    predicted_labels = np.argmax(batch_predictions, axis=1)

    # 计算正确预测的数量
    correct_prediction += np.sum(predicted_labels == batch_labels)

accuracy = correct_prediction / num_sample
print("the model's accuracy: {:.2%}".format(accuracy))
