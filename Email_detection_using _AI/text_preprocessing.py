"""
UTILS.PY — Shared Utility Functions

Contains:
1. clean_text() : Cleans raw email text using the SAME steps
Used in both training and prediction for consistency.
"""

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Clean text by removing punctuation, lowercasing, removing stopwords, and lemmatizing."""
    
    # Keep only letters
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Lowercase
    text = text.lower()

    # Split into tokens
    words = text.split()

    # Remove stopwords + lemmatize
    filtered_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return ' '.join(filtered_words)

