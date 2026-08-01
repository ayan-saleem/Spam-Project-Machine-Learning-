"""
Spam Email Classifier
----------------------
A simple text-classification project using:
  - TF-IDF for turning email text into numeric features
  - Multinomial Naive Bayes for classification 

Run:
    python SpamClassifier.py


    
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
# A small hand-crafted dataset so the script runs out of the box.
# I will swap this out for a CSV (e.g. the classic "SMS Spam
# Collection" dataset) using: df = pd.read_csv("spam.csv")
def load_sample_data() -> pd.DataFrame:
    emails = [
        # --- Spam ---
        ("Congratulations! You've won a $1000 Walmart gift card. Click here now!", "spam"),
        ("URGENT: Your account has been suspended. Verify your details immediately.", "spam"),
        ("Get cheap meds online, no prescription needed! Buy now and save 80%.", "spam"),
        ("You have been selected for a free cruise! Reply YES to claim.", "spam"),
        ("Make $5000 a week working from home, no experience required!", "spam"),
        ("Lowest interest rates guaranteed! Apply for your loan today.", "spam"),
        ("Hot singles in your area want to meet you tonight!", "spam"),
        ("Claim your free iPhone now, limited time offer, click the link!", "spam"),
        ("Your PayPal account needs verification, click here to avoid suspension.", "spam"),
        ("WINNER!! As a valued customer you have been selected to receive a prize.", "spam"),
        ("Increase your credit score instantly, guaranteed results!", "spam"),
        ("Free entry to win a brand new car, text WIN to 80085 now.", "spam"),
        ("Enlarge your... business opportunity! Work from home and get rich fast.", "spam"),
        ("Your loan has been pre-approved, click to receive funds today!", "spam"),
        ("Act now! This offer expires in 24 hours, don't miss out!", "spam"),
        # --- Ham (not spam) ---
        ("Hey, are we still on for lunch tomorrow at noon?", "ham"),
        ("Please find attached the quarterly report for your review.", "ham"),
        ("Can you send me the notes from yesterday's meeting?", "ham"),
        ("Happy birthday! Hope you have a wonderful day.", "ham"),
        ("The flight has been delayed by two hours, see you at the gate.", "ham"),
        ("Reminder: your dentist appointment is scheduled for 3pm on Friday.", "ham"),
        ("Thanks for the update, I'll review the document tonight.", "ham"),
        ("Let's catch up this weekend, it's been a while!", "ham"),
        ("The project deadline has been moved to next Monday.", "ham"),
        ("Could you please review my pull request when you get a chance?", "ham"),
        ("Mom, I'll be home late tonight, don't wait up for dinner.", "ham"),
        ("Here's the recipe you asked for, let me know how it turns out.", "ham"),
        ("Our team meeting is rescheduled to 10am tomorrow.", "ham"),
        ("I really enjoyed the movie we watched last night.", "ham"),
        ("Attached is the invoice for last month's services.", "ham"),
    ]
    return pd.DataFrame(emails, columns=["text", "label"])


# ---------------------------------------------------------------------------
# 2. TRAIN
# ---------------------------------------------------------------------------
def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    # Convert text into TF-IDF feature vectors
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate
    predictions = model.predict(X_test_vec)
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))
    print()

    return model, vectorizer


# ---------------------------------------------------------------------------
# 3. PREDICT
# ---------------------------------------------------------------------------
def predict_email(model, vectorizer, text: str):
    vec = vectorizer.transform([text])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    confidence = dict(zip(classes, proba))
    return label, confidence


# ---------------------------------------------------------------------------
# 4. MAIN / DEMO
# ---------------------------------------------------------------------------
def main():
    df = load_sample_data()
    model, vectorizer = train_model(df)

    # Try it on some new, unseen examples
    test_emails = [
        "Click here to claim your free prize now, limited time only!",
        "Hey, can we reschedule our call to Thursday afternoon?",
        "Congratulations, you've been chosen to win $10,000, act fast!",
        "Attached is the file you requested, let me know if you need anything else.",
    ]

    print("=" * 50)
    print("PREDICTIONS ON NEW EMAILS")
    print("=" * 50)
    for email in test_emails:
        label, confidence = predict_email(model, vectorizer, email)
        print(f"\nEmail: \"{email}\"")
        print(f"Prediction: {label.upper()}")
        print(f"Confidence -> ham: {confidence['ham']:.2%}, spam: {confidence['spam']:.2%}")

    # Optional: interactive mode
    print("\n" + "=" * 50)
    print("Try your own email (press Enter with no text to quit)")
    print("=" * 50)
    while True:
        user_input = input("\nEnter an email to classify: ").strip()
        if not user_input:
            print("Goodbye!")
            break
        label, confidence = predict_email(model, vectorizer, user_input)
        print(f"-> Prediction: {label.upper()} "
              f"(ham: {confidence['ham']:.2%}, spam: {confidence['spam']:.2%})")


if __name__ == "__main__":
    main()
