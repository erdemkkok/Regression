#!/usr/bin/(env python3

import cv2
import gi
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()  #Model Yüklemesi yapılmaktadır.


x_train, x_test = x_train / 255.0, x_test / 255.0        #Derin öğrenme verileri sadece 0 ve 1 arası yapıda çalıştığı için 255 e bölme işlemi yapılmaktadır.
print(x_train.shape)

model = tf.keras.models.Sequential([                        #Model mimarisi oluşturulmaktadır.
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10,activation='softmax')
])

predictions = model(x_train[:1]).numpy()

print("Predictions",predictions)
# # # predictions

# tf.nn.softmax(predictions).numpy()
# print("Softmax_output",tf.nn.softmax(predictions).numpy())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


loss_fn(y_train[:1], predictions).numpy()

# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)


# model.evaluate(x_test,  y_test, verbose=2)

# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])
"""
Yukarida kullanilmamasinin sebebi - verilerinde öğrenilebilmesidir. 
Bu kullanim ile yeniden bir sequential model kullanilarak - veriler de öğrenimde hesaba katilmiştir.
Bu da bize daha iyi bir tahmin saglamaktadir.
"""
# print(probability_model(x_test[:5]))
