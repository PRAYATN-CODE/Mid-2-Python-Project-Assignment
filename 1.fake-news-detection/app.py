import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up page configuration
st.set_page_config(page_title="Global News Verifier", layout="centered")

# Custom CSS targeting your specific UI design requirements
st.markdown("""
    <style>
    div.stButton > button {
        border-radius: 0.75rem; /* rounded-xl */
        border: 1px solid #e5e7eb; /* border-gray-200 */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
        background-color: #ffffff;
        color: #374151;
        transition: all 0.2s;
        padding: 0.5rem 1rem;
    }
    div.stButton > button:hover {
        border-color: #d1d5db;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div.stTextArea > div > textarea {
        border-radius: 0.75rem; /* rounded-xl */
        border: 1px solid #e5e7eb; /* border-gray-200 */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    # Load the unified pipeline file
    with open('fake_news_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

pipeline_model = load_model()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'^.*? - ', '', str(text)) 
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

# App UI
st.title("📰 Global News Verification Engine")
st.write("Analyze reports on global equity markets, Bitcoin, Ethereum, and international events.")

# User Input
user_input = st.text_area("Article Text:", height=200, placeholder="Paste international news content here...")

if st.button("Analyze Article"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text patterns..."):
            cleaned_text = preprocess_text(user_input)
            
            # The pipeline automatically vectorizes and predicts in one step
            prediction = pipeline_model.predict([cleaned_text])[0]
            decision_score = pipeline_model.decision_function([cleaned_text])[0]

            st.markdown("### Results:")
            if prediction == 1:
                st.error(f"🚨 **Flagged as Fake News** (Algorithm Score: {decision_score:.2f})")
            else:
                st.success(f"✅ **Appears to be Real News** (Algorithm Score: {decision_score:.2f})")