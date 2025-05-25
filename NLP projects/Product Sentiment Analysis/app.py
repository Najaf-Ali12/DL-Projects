## Save this as app.py
#
#import streamlit as st
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#import numpy as np
#
## Load your trained model
#model = load_model('sentiment_model.keras')
#
## Define constants (should match training settings)
#voc_size = 1000
#max_sentence_length = 20
#
## Function to one-hot encode the text (same logic as you used)
#def one_hot_encode(text, voc_size):
#    from tensorflow.keras.preprocessing.text import one_hot
#    return one_hot(text, voc_size)
#
## Mapping from prediction to sentiment label
#sentiment_mapping = {
#    0: "Cannot say",
#    1: "Negative",
#    2: "Positive",
#    3: "No sentiment"
#}
#
## Streamlit Frontend
#st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")
#
#st.title("💬 Sentiment Analysis App")
#
#st.write("### Enter your review below:")
#
## User input
#user_input = st.text_area("Your Review:", height=150)
#
#if st.button("Analyze Sentiment"):
#    if user_input.strip() == "":
#        st.warning("Please enter some text for analysis.")
#    else:
#        # Preprocess input
#        encoded_input = one_hot_encode(user_input, voc_size)
#        padded_input = pad_sequences([encoded_input], maxlen=max_sentence_length, padding='pre')
#
#        # Predict
#        y_pred = model.predict(padded_input)
#        predicted_class = np.argmax(y_pred, axis=1)[0]
#
#        # Show result
#        st.success(f"🌟 Sentiment: **{sentiment_mapping[predicted_class]}**")
#
#st.markdown("---")
#st.markdown("<h5 style='text-align: center; color: grey;'>Made with ❤️ by Najaf Ali</h5>", unsafe_allow_html=True)
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('sentiment_model.keras')

# Vocabulary size and max sentence length should match the training settings
voc_size = 5000
max_sentence_length = 20

#one_hot encoder function
def one_hot_encode(text):
    from tensorflow.keras.preprocessing.text import one_hot
    return one_hot(text, voc_size)

# Load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply CSS
local_css('style.css')

# App Title
st.title("Sentiment Analysis App")
st.subheader("Developed by Najaf Ali 🚀")

# User Input
review = st.text_area("Enter your Review Here:", "")

# Predict Button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        with st.spinner('Analyzing... Please wait ⏳'):
            # Preprocess
            encoded_review = one_hot_encode(review)
            encoded_review = np.expand_dims(encoded_review, axis=0)  # Model expects batch dimension
            
            # Predict
            y_pred = model.predict(encoded_review)
            predicted_class = np.argmax(y_pred, axis=1)[0]
            confidence = np.max(y_pred) * 100
            
            # Mapping classes
            classes = {
                0: "Cannot say",
                1: "Negative",
                2: "Positive",
                3: "No Sentiment"
            }
            
            # Show result
            st.success(f"Predicted Sentiment: **{classes[predicted_class]}** 🎯")
            st.info(f"Model Confidence: {confidence:.2f}%")

# Footer
st.markdown("---")
st.caption("© 2025 Najaf Ali. All rights reserved.")
