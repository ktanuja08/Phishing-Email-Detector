import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

print("Loading dataset...")

# Load dataset
df = pd.read_csv("dataset.csv")

# Select required columns
df = df[['Email Text', 'Email Type']]

# Convert labels to numeric
df['Email Type'] = df['Email Type'].map({
    'Safe Email': 0,
    'Phishing Email': 1
})

print("Dataset loaded successfully!\n")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)   # detect URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

# Apply cleaning
df['Email Text'] = df['Email Text'].apply(clean_text)

# Features and labels
X = df['Email Text']
y = df['Email Type']

# Convert text → numbers
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained successfully!\n")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Function to predict new email
def predict_email(email):
    email = clean_text(email)
    email_vec = vectorizer.transform([email])
    result = model.predict(email_vec)[0]
    return "Phishing" if result == 1 else "Safe"


# 🔹 Interactive loop
while True:
    print("\nEnter an email message (or type 'exit' to quit):")
    user_email = input()

    if user_email.lower() == 'exit':
        print("Exiting program...")
        break

    prediction = predict_email(user_email)
    print("Prediction:", prediction)
