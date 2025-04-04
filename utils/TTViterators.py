# Train test split
import tensorflow as tf
# import os
# import numpy as np
# from tqdm import tqdm 
# from skimage.io import imread, imshow
#from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
# import matplotlib.pyplot as plt


def get_train_test_val_iterators(
    X, Y, 
    batch_size=20, 
    train_test_ratio=0.2, 
    train_val_ratio=0.1, 
    train_test_seed=42, 
    train_val_seed=42):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=train_test_ratio, random_state=train_test_seed)
    print(X_train.shape==Y_train.shape, X_train.shape)
    del X, Y
    print(X_train.shape, Y_train.shape, X_train.dtype, Y_train.dtype, type(X_train), type(Y_train), X_test.shape, Y_test.shape, X_test.dtype, Y_test.dtype, type(X_test), type(Y_test))


    # Validation split
    # from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=train_val_ratio, random_state=train_val_seed) # 0.25 x 0.8 = 0.2
    print('Train ', X_train.shape==Y_train.shape, X_train.shape, X_train.dtype==Y_train.dtype, X_train.dtype)
    print('Val   ', X_val.shape==Y_val.shape, X_val.shape, X_val.dtype==Y_val.dtype, X_val.dtype)
    print('Test. ', X_test.shape==Y_test.shape, X_test.shape, X_test.dtype==Y_test.dtype, X_test.dtype)



    # Normalization (assumned inside model)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(...)
    datagen.fit(X_train)



    ## new categorical
    # Assuming Y_train is a 1D array containing class labels
    # Convert Y_train to categorical format
    Y_train_categorical = to_categorical(Y_train)
    del Y_train
    Y_val_categorical = to_categorical(Y_val)
    del Y_val
    Y_test_categorical = to_categorical(Y_test)
    del Y_test

    # batch_size = 20
    train_iterator = datagen.flow(X_train, Y_train_categorical, batch_size=batch_size)
    val_iterator = datagen.flow(X_val, Y_val_categorical, batch_size=batch_size)
    test_iterator = datagen.flow(X_test, Y_test_categorical, batch_size=batch_size)
    # del to_categorical

    return train_iterator, test_iterator, val_iterator

# train_iterator, test_iterator, val_iterator = get_train_test_val_iterators(X,Y)