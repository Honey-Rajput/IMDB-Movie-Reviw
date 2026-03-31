import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import load_model
import streamlit as st
# Load the IMDB dataset word index

word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

#Load the pre-trained model with Relu activation function
model = load_model('simple_rnn_relu_model.h5')

# Function to preprocess user input

def preprocess_input(text):
    words = text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


### Prediction Function

#def predict_sentiment(review):
   # preprocessed_input = preprocess_input(review)
   # prediction = model.predict(preprocessed_input)
   # sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
   # return sentiment,prediction[0][0]  


### Streamlit App

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

## User input
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    preprocessed_input=preprocess_input(user_input)

    ##Make Predition
    prediction=model.predict(preprocessed_input)
    sentiment="Positive" if prediction[0][0]>0.5 else "Negative"
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.2f}")


else:
    st.write("Please enter a movie review ")