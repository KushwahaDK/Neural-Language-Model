from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    """ Function to generate a sequence of characters with a language model

    Args:
        model (keras model): trained keras model
        mapping (dict): dictionary mapping of all unique input charcters to integers
        seq_length (integer): length of the input sequence
        seed_text (string): the starting text
        n_chars (integer): number of characters to be predcited

    Returns:
        string: returns starting text + predicted text
    """
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict character
        yhat = np.argmax(model.predict(encoded), axis=-1)
        # reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # append to input
        in_text += char
    return in_text


# load Keras trained model
model = load_model('models/char_based_neural_lang_model.h5')

# load mapping pickle file
mapping = pickle.load(open('models/mapping.pickle', 'rb' ))

# input sequence length
sequence_length = 30

# inital text for prediction
seed_text = 'his autobiography'

# next number of characters for prediction
prediction_chars = 50

# final generated text using language models
generated_text = generate_seq(model, mapping, sequence_length, seed_text, prediction_chars)

print('complete sentence generated is : {}'.format(generated_text))