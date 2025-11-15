"""
EVALUATE SCRIPT — Load and Predict

Loads:
- spam_model.pkl
- vectorizer.pkl

Steps:
1. Clean the new email using clean_text() from utils.py
2. Convert into vector using SAME vectorizer
3. Predict using trained model
"""

import pickle
from text_preprocessing import clean_text  # <- SAME CLEAN FUNCTION


# Load model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)


def predict_email(email_text):
    """Return Spam / Not Spam prediction for a new email."""
    
    cleaned = clean_text(email_text)
    transformed = cv.transform([cleaned])
    pred = model.predict(transformed)[0]

    return "Spam" if pred == 1 else "Not Spam"


if __name__ == "__main__":

    test_email = "Hi Team, please find the attached report and review it."

    result = predict_email(test_email)

    print("\n----- EMAIL PREDICTION -----")
    print("Email:", test_email)
    print("Prediction:", result)
