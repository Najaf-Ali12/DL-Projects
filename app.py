import streamlit as st

import joblib
import base64
import re # Regular expression for text preprocessing



# Setting The name of website
st.set_page_config(page_title="Emotion Detector App", page_icon=":guardsman:", layout="wide")


# Function to encode an image file to a base64 string for use as a background
def get_base64_image(image_path):
    """
    Encodes an image file to a base64 string.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to your image file
image_file_path = "emotion detector background image.jpeg"

# Get the base64 string of your image
try:
    encoded_image = get_base64_image(image_file_path)
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"Error: Background image '{image_file_path}' not found. Please ensure it's in the same directory.")
    # You might want to continue without background or stop the app here
except Exception as e:
    st.error(f"An error occurred while loading the background image: {e}")




# Loading all necessary components

## Loading the tf-idf vectorizer we used
tf_idf=joblib.load("TF-IDF Vectorizer of Emotion Detector using Random Forest.joblib")



## Loading the Label_Encoder we used
encoder=joblib.load("Label Encoder of Emotion Detector using Random Forest.joblib")



# Loading the model that we trained
model=joblib.load("Emotion Detector using Random Forest.joblib")



# Title of the app
st.title("Emotion Detector App",anchor="NLP Project")



# Sidebar of app for developer identity and Porject Repository

st.sidebar.header("Developer Identity and Project Repository")

st.sidebar.page_link("https://github.com/Najaf-Ali12/DL-Projects",label="Project Repository Link",icon="🔥")

st.sidebar.info("This project is developed by Najaf Ali, a Data Scientist and Machine Learning Engineer of Future. You can find more about me on my [LinkedIn](https://www.linkedin.com/in/najaf-ali12/).")



# Input text from user

text=st.text_input("Enter your text here to detect emotion: ",key="Input_Text")

if not text:

    st.warning("Please enter some text to detect emotion.") # Warning if no text is entered

else:

    if st.button("Detect Emotion"):

        text=re.sub("[^a-zA-Z!?]"," ", text) # Preprocessing the text to remove non-alphabetic characters

        text=text.lower() # Converting text to lowercase

        text=tf_idf.transform([text]) # Transforming the text using the tf-idf vectorizer

        prediction=model.predict(text) # Making prediction using the trained model

        emotion=encoder.inverse_transform(prediction) # Inverse transforming the prediction to get the original emotion label

        st.success(f"The emotion detected is: {emotion[0]}") # Displaying the detected emotion

    else:

        st.info("Click the button to detect emotion from the input text.")


# Footer section for feedback and rating
st.markdown("---")
st.header("💖 Provide Your Feedback")
st.write("We'd love to hear what you think about the app!")

rating_options = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
selected_rating = st.selectbox(
    "How would you rate this app?",
    options=rating_options,
    index=2,
    help="Select your rating from 1 to 5 stars."
)

user_comments = st.text_area(
    "Your Comments (Optional):",
    placeholder="Share your thoughts, suggestions, or any issues you encountered...",
    height=150
)

if st.button("Submit Feedback"):
    st.success("🎉 Thank you for your feedback! Your input is valuable.")