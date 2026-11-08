"""
Age estimation from face photos.

Trains a ResNet50 regression model on the ChaLearn Looking at People faces
dataset and prints MAE per epoch. Run this on a GPU machine: 7,591 images
through ResNet50 for 20 epochs is not a CPU job.

    python run_model_on_gpu.py
"""

import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


def load_train(path):

    """
    It loads the train part of dataset from path
    """

    labels = pd.read_csv(path + 'labels.csv')

    train_datagen = ImageDataGenerator(
        rescale=1/255.,
        horizontal_flip=True,
        validation_split=0.25,
    )

    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345,
    )

    return train_gen_flow


def load_test(path):

    """
    It loads the validation/test part of dataset from path
    """

    labels = pd.read_csv(path + 'labels.csv')

    test_datagen = ImageDataGenerator(
        rescale=1/255.,
        validation_split=0.25,
    )

    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345,
    )

    return test_gen_flow


def create_model(input_shape):

    """
    It defines the model
    """

    backbone = ResNet50(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
    )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae'],
    )

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    """
    Trains the model given the parameters
    """

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)

    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
    )

    return model


if __name__ == '__main__':

    PATH = '/datasets/faces/'

    train_data = load_train(PATH)
    test_data = load_test(PATH)

    model = create_model(input_shape=(224, 224, 3))
    model = train_model(model, train_data, test_data)
