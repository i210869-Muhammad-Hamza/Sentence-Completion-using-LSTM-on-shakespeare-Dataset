#libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json

# Load the variables from the JSON file
with open('variables.json', 'r') as f:
    variables = json.load(f)

# Load the trained model
@st.cache_resource
def load_lstm_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function to predict the next word
def predict_next_word(model, user_input, word_to_index, index_to_word, seq_length):
    # Convert user input into tokens (integers)
    tokenized_input = [word_to_index[word] for word in user_input if word in word_to_index]
    
    # Pad or truncate input to the required sequence length
    tokenized_input = pad_sequences([tokenized_input], maxlen=seq_length, padding='pre')
    
    # Predict the next word (returns probabilities for each word in the vocab)
    predicted_probabilities = model.predict(tokenized_input, verbose=0)[0]
    
    # Get the word index with the highest probability
    predicted_index = np.argmax(predicted_probabilities)
    
    # Return the corresponding word
    return index_to_word[predicted_index]

def predict_next_n_words(model, user_input, vocab, index_to_word, seq_length, n_words=4):
    # Ensure user_input is a list of strings
    if isinstance(user_input, str):
        user_input = user_input.split()  # Split string into a list of words

    predicted_words = []

    for _ in range(n_words):
        # Predict the next word
        predicted_word = predict_next_word(model, user_input, vocab, index_to_word, seq_length)

        # Append the predicted word to the result
        predicted_words.append(predicted_word)

        # Update the input by shifting and appending the predicted word
        user_input = user_input[1:] + [predicted_word]

    return predicted_words


# Initialize app
st.title("Next Word Prediction with LSTM")
st.write("Enter your text")

# User input
user_input = st.text_input("Enter 5 words:")


import streamlit as st

# Load the LSTM model
model_path = 'lstm_model.h5'  # Replace with your trained model path
lstm_model = load_lstm_model(model_path)

seq_length = 5  # Assuming you're using 5-word sequences
index_to_word = {v: k for k, v in variables['vocab'].items()}

# Initialize session state for input text
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Input box for user input
user_input = st.text_area("Enter your text:", st.session_state.input_text)

# Update session state if input changes
if user_input != st.session_state.input_text:
    st.session_state.input_text = user_input  # Update session state

# Display current input
st.write(f"Current input: {st.session_state.input_text}")

# Split input into words
input_words = st.session_state.input_text.split()

# Ensure there are at least 5 words for prediction
if len(input_words) >= 5:
    if st.button("Predict Next Words"):
        # Get the last 5 words for prediction
        last_five_words = input_words[-5:]  # Take the last 5 words
        predicted_words = predict_next_n_words(lstm_model, last_five_words, variables['vocab'], index_to_word, seq_length)

        # Update the input text with the predicted words
        st.session_state.input_text += " " + " ".join(predicted_words)
        st.session_state.input_text = st.session_state.input_text.strip()  # Clean up any extra spaces

        # Update the text area to show the updated input
        st.text_area("Enter your text:", st.session_state.input_text, height=200)

        # Show the predicted words
        st.write(f"Next words predicted: {', '.join(predicted_words)}")
else:
    st.write("Please enter at least 5 words to make a prediction.")
