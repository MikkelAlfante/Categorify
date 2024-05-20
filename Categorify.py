import streamlit as st
import pandas as pd
import pickle


# Set up the page configuration
st.set_page_config(
    page_title="Categorify",
    page_icon="🔖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Streamlit app
st.title("Categorify: Text Categorization App")
st.write("Enter text to categorize into predefined categories.")

# Load the model
with open('text_classifier_bbc.pkl', 'rb') as file:
    model = pickle.load(file)

# Text input
user_input = st.text_area("Enter your text here:")

# Predict category
if st.button("Categorize"):
    if user_input:
        prediction = model.predict([user_input])
        st.write(f"Predicted Category: **{prediction[0]}**")

        # Show category probabilities
        probabilities = model.predict_proba([user_input])
        st.write("Category Probabilities:")
        prob_df = pd.DataFrame(probabilities, columns=model.classes_)
        st.write(prob_df.T)
    else:
        st.write("Please enter some text to categorize.")

# Allow users to upload their own datasets
st.header("Upload Your Own Dataset")
st.write("Please upload a CSV file with a column named 'text' containing the text data to categorize.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.write(user_data.head())
    
    if 'text' in user_data.columns:
        if st.button("Categorize Uploaded Data"):
            user_X = user_data['text']
            user_predictions = model.predict(user_X)
            user_data['predicted_category'] = user_predictions
            st.write("Predicted Categories for Uploaded Data:")
            st.write(user_data)
    else:
        st.error("Uploaded CSV does not contain the required 'text' column.")

    
