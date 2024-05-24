import streamlit as st
import pandas as pd
import openai
import pickle
import time
from openai.error import OpenAIError, RateLimitError

try:
    import openai
    from openai.error import OpenAIError, RateLimitError
except ImportError as e:
    st.error(f"Error importing OpenAI package: {e}. Please ensure it is installed.")

# Set up the page configuration
st.set_page_config(
    page_title="Categorify",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🔖"
)

# Function to load external HTML file
def load_html(file_name):
    with open(file_name) as f:
        html_content = f.read()
        st.markdown(html_content, unsafe_allow_html=True)

# Function to get assessment from GPT-3.5 Turbo
def get_gpt3_5_turbo_assessment(text, category, api_key):
    openai.api_key = api_key
    try:
        prompt = f"The following text was classified into the category '{category}'. Is this classification correct? Explain why or why not.\n\nText: {text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['message']['content'].strip()
    except RateLimitError:
        st.error("You have exceeded your API quota. Please check your OpenAI plan and billing details.")
        return None
    except OpenAIError as e:
        st.error(f"An error occurred: {e}")
        return None

# Load the pre-trained model
with open('text_classifier_bbc.pkl', 'rb') as file:
    model = pickle.load(file)

# Sidebar for API key input
st.sidebar.title("API Key Required")
api_key_input = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

# Validate API key
if st.sidebar.button("Confirm API Key"):
    if api_key_input:
        with st.spinner("Validating API key..."):
            try:
                openai.api_key = api_key_input
                openai.Model.list()  # Perform a simple API call to validate the key
                st.sidebar.success("API key validated successfully!")
                st.session_state.api_key_confirmed = True
                st.session_state.api_key_input = api_key_input  # Store the API key
            except OpenAIError as e:
                st.sidebar.error(f"Invalid API key: {e}")
                st.session_state.api_key_confirmed = False
    else:
        st.sidebar.warning("Please enter your OpenAI API key.")
        st.session_state.api_key_confirmed = False

# Check if the API key is confirmed
if st.session_state.get('api_key_confirmed'):
    # Page Title
    st.title("Categorify: Text Categorization App")

    # Introductory message
    st.write("Welcome to Categorify, your tool for text categorization. Enter text to classify it into predefined categories using a pre-trained model and get an assessment from GPT-3.5 Turbo.")

    # Text input
    st.subheader("Enter your text here:")
    user_input = st.text_area("Text Input", key="text_input", height=200, help="Type or paste your text here.", label_visibility="collapsed")

    # Container for the loading GIF
    loading_container = st.empty()

    # Predict category
    if st.button("Categorize"):
        if user_input:
            with st.spinner('Categorizing...'):
                loading_container.image("loading.gif")  # Display the loading GIF
                # Pre-trained model prediction
                pre_trained_prediction = model.predict([user_input])[0]
                # ChatGPT assessment
                chatgpt_assessment = get_gpt3_5_turbo_assessment(user_input, pre_trained_prediction, st.session_state.api_key_input)

                # Clear the loading GIF
                loading_container.empty()

                # Display predictions and assessment
                st.subheader(f"Pre-trained Model Prediction: {pre_trained_prediction}")
                if chatgpt_assessment:
                    st.subheader("ChatGPT Assessment:")
                    st.write(chatgpt_assessment)
        else:
            st.warning("Please enter some text to categorize.")

    # Allow users to upload their own datasets
    st.subheader("Upload Your Own Dataset")
    st.write("Please upload a CSV file with a column named 'text' containing the text data to categorize.")
    uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.subheader("Preview of uploaded data")
        st.write(user_data.head())

        if 'text' in user_data.columns:
            if st.button("Categorize Uploaded Data"):
                with st.spinner('Categorizing...'):
                    loading_container.image("loading.gif")  # Display the loading GIF
                    # Pre-trained model predictions
                    user_data['pre_trained_category'] = user_data['text'].apply(lambda x: model.predict([x])[0])
                    # ChatGPT assessments
                    user_data['chatgpt_assessment'] = user_data.apply(
                        lambda row: get_gpt3_5_turbo_assessment(row['text'], row['pre_trained_category'], st.session_state.api_key_input),
                        axis=1
                    )

                    # Clear the loading GIF
                    loading_container.empty()

                    st.subheader("Categorized Data")
                    st.write(user_data)
                    csv = user_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Categorized Data",
                        data=csv,
                        file_name='categorized_data.csv',
                        mime='text/csv'
                    )
        else:
            st.error("Uploaded CSV does not contain the required 'text' column.")
else:
    st.warning("Please enter and confirm your OpenAI API key to use the app.")

# Load footer HTML
load_html("footer.html")