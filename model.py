import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import os
import math
import tensorflow_datasets as tfds

directory = '/train'

classNames = []
with os.scandir(directory) as entries:
    for entry in entries:
        if entry.is_dir():
            classNames.append(entry.name)

seed = 72
random.seed(seed)

trainDataset = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    class_names=classNames,
    color_mode='rgb',
    batch_size=64,
    image_size=(224, 224),
    shuffle=True,
    seed=seed,
    validation_split=0.15,
    subset='training',
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

seed = 172
random.seed(seed)

valDataset = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    class_names=classNames,
    color_mode='rgb',
    batch_size=64,
    image_size=(224, 224),
    shuffle=True,
    seed=seed,
    validation_split=0.15,
    subset='validation',
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)


dataAugmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomBrightness(0.1),
    keras.layers.RandomContrast(0.1),
])

augDataset = trainDataset.map(
    lambda x, y: (dataAugmentation(x, training=True), y))

combinedDataset = trainDataset.concatenate(augDataset)
combinedDataset = combinedDataset.shuffle(buffer_size=21600)

baseModel = tf.keras.applications.EfficientNetB7(input_shape=(224, 224, 3),
                                                 include_top=False,
                                                 weights='imagenet')

for layer in baseModel.layers:
    layer.trainable = False

x = baseModel.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(len(classNames), activation='softmax')(x)
output_layer = x
model = tf.keras.models.Model(baseModel.input, output_layer)


def stepDecay(epoch):
    initialLr = 0.01
    drop = 0.5
    epochsDrop = 10.0
    lr = initialLr * math.pow(drop, math.floor((1+epoch)/epochsDrop))
    return lr


lrScheduler = tf.keras.callbacks.LearningRateScheduler(stepDecay)

optimizer = tf.keras.optimizers.SGD(momentum=0.9)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

history = model.fit(combinedDataset,
                    epochs=40,
                    validation_data=valDataset,
                    shuffle=True,
                    callbacks=[lrScheduler]
                    )

model.save('/trainedModel')
