from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import optimizers
import tensorflow as tf

from data_batching import train_generator
from data_batching import valid_generator


conv_model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),

    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')
])

conv_model.compile(
  loss='binary_crossentropy',
  optimizer=optimizers.RMSprop(lr=1e-4),
  metrics=['acc'])

with tf.device('/device:GPU:0'):
  history = conv_model.fit(
    train_generator, steps_per_epoch=11,
    validation_data=valid_generator,
    validation_steps=6,
    epochs=30)

conv_model.save('model.h5')