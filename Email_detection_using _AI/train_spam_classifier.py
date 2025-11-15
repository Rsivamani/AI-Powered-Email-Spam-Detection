"""
TRAINING SCRIPT — SPAM CLASSIFIER

Steps:
1. Load dataset
2. Clean text using clean_text() from utils.py
3. Convert to features using CountVectorizer
4. Train Naive Bayes classifier
5. Save model + vectorizer using pickle
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from text_preprocessing import clean_text   # <- SHARED CLEANING FUNCTION


def train_model():

    # Load dataset
    data = pd.read_csv('email_dataset.tsv', sep='\t', names=['label', 'messages'])

    # Clean all emails
    cleaned_emails = [clean_text(msg) for msg in data['messages']]

    # Vectorize
    cv = CountVectorizer(max_features=3500, stop_words='english')
    X = cv.fit_transform(cleaned_emails).toarray()

    # Labels: ham = 0, spam = 1
    y = pd.get_dummies(data['label'], drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

    # Train model
    model = MultinomialNB(alpha=0.8)
    model.fit(X_train, y_train)

    # Save model + vectorizer
    with open("./model/spam_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("./model/vectorizer.pkl", "wb") as f:
        pickle.dump(cv, f)

    print("MODEL TRAINED & SAVED SUCCESSFULLY!")


if __name__ == "__main__":
    train_model()
