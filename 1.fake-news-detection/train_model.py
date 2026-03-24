import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Download required NLTK data silently
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Strip standard publisher tags to prevent data leakage (the "Reuters" cheat)
    text = re.sub(r'^.*? - ', '', str(text)) 
    # Keep only letters, convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    # Lemmatize and remove stopwords
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

if __name__ == "__main__":
    print("Loading datasets... (Ensure Fake.csv and True.csv are in this folder)")
    
    # Load the data
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')

    # Assign labels: 1 for Fake News, 0 for True News
    fake_df['label'] = 1
    true_df['label'] = 0

    # Combine and shuffle the datasets
    df = pd.concat([fake_df, true_df], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Training on {len(df)} articles. Preprocessing text... (This takes a few minutes)")
    X = df['text'].apply(preprocess_text)
    y = df['label']

    # Split into 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building and training the final Pipeline...")
    
    # The Pipeline handles both TF-IDF vectorization (with n-grams) and the Passive Aggressive Classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))),
        ('pac', PassiveAggressiveClassifier(max_iter=50))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Test the accuracy on the 20% holdout set
    y_pred = pipeline.predict(X_test)
    print(f"\nFinal Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save as a single, unified file for the Streamlit app to use
    with open('fake_news_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
    print("\nSuccess! Saved as 'fake_news_pipeline.pkl'")