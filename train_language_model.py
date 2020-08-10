# Program to Create a Neural language Model

# required imports
import os
import re
import warnings
import numpy as np
import pandas as pd
import pickle

from nltk import ngrams
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense


warnings.simplefilter(action='ignore', category=FutureWarning)


def read_text(path):
    """ Function to read input data

    Args:
        path (string): the parent path of the folder containing the input text files

    Returns:
        string: The complete text read from input files appended in a single string.
    """
    text = ' '
    for text_file in os.listdir(path):
        text_file_buffer = open(os.path.join(
            path, text_file), mode='r', encoding='utf-8')
        text = text + text_file_buffer.read()
        text_file_buffer.close()
    # returning first 100000 reviews only because dataset is too large for training on a CPU.
    return text[:100000]


def preprocess_text(text):
    """ Function for basic cleaning and pre-processing of input text

    Args:
        text (string): raw input text

    Returns:
        string: cleaned text
    """
    text = text.lower()
    text = re.sub(r"'s\b", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join([word for word in text.split() if len(word) >= 3]).strip()

    return text


def prepare_text(text):
    """ Function to prepare text in sequence of ngrams

    Args:
        text (string): complete input text

    Returns:
        list : a list of text sequence with 31 characters each
    """
    sequence = list(ngrams(text, 31))
    sequence = [''.join(char for char in sub_sequence)
                for sub_sequence in sequence]
    return sequence


def encoding_sequence(text, sequence):
    """ Function to encode the character sequence into number sequence

    Args:
        text (string): cleaned text
        sequence (list): character sequence list

    Returns:
        dict: dictionary mapping of all unique input charcters to integers
        list: number encoded charachter sequences
    """

    mapping = dict((c, i) for i, c in enumerate(sorted(list(set(text)))))
    encoded_sequence = [[mapping[char]
                         for char in sub_sequence] for sub_sequence in sequence]

    return mapping, encoded_sequence


def split_data(mapping, encoded_sequence):
    """ Function to split the prepared data in train and test

    Args:
        mapping (dict): dictionary mapping of all unique input charcters to integers
        encoded_sequence (list): number encoded charachter sequences

    Returns:
        numpy array : train and test split numpy arrays
    """

    encoded_sequence_ = np.array(encoded_sequence)
    X, y = encoded_sequence_[:, :-1], encoded_sequence_[:, -1]
    y = to_categorical(y, num_classes=len(mapping))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(mapping, X_train, X_test, y_train, y_test):
    """Function to train the model

    Args:
        mapping (dict): dictionary mapping of all unique input charcters to integers
        X_train (ndarray): Training data feature split
        X_test (ndarray): Testing data feature split
        y_train (ndarray): training data target split
        y_test (ndarray): testing data target split

    Returns:
        keras model : Trained keras model
    """
    vocab = len(mapping)

    model = Sequential()
    model.add(Embedding(vocab, 50, input_length=30, trainable=True))
    model.add(GRU(50, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocab, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print("\n")
    print(" !!Fitting the model !! ")
    model.fit(X_train, y_train, epochs=20, verbose=2,
              validation_data=(X_test, y_test))

    return model


def save_model(mapping, model, path):
    """ Function to save mapping and trained model

    Args:
        mapping (dict): dictionary mapping of all unique input charcters to integers
        model (keras model): trained keras model
        path (string): path where model will be stored
    """

    model.save(path)

    pickle.dump(mapping, open('models/mapping.pickle', 'wb'))


if __name__ == "__main__":

    path = 'imdb_data/'

    print(" !!.... Reading texts ....!! ")
    text = read_text(path)

    print(" !!.... Preprocessing texts ....!! ")
    text = preprocess_text(text)

    print(" !!.... Preparing texts ....!! ")
    sequence = prepare_text(text)

    print(" !!.... Encode sequence ....!! ")
    mapping, encoded_sequence = encoding_sequence(text, sequence)

    print(" !!.... Spliting data ....!! ")
    X_train, X_test, y_train, y_test = split_data(
        mapping, encoded_sequence)

    print(" !!.... Training model ....!! ")
    model = train_model(mapping, X_train, X_test, y_train, y_test)

    print(" !!.... Saving model and Mapping ....!! ")
    model_path = 'models/char_based_neural_lang_model.h5'
    save_model(mapping, model, model_path)
