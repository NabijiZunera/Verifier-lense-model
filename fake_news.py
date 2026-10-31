
import re
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "dataset" / "train.csv"

MODEL_FILE = BASE_DIR / "fake_news_model.pkl"
VECTORIZER_FILE = BASE_DIR / "tfidf_vectorizer.pkl"


# ---------------------------------------
# 2. LOAD DATASET
# ---------------------------------------

print("\nLoading training dataset...")

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_FILE}"
    )

train_data = pd.read_csv(TRAIN_FILE)

required_columns = ["title", "text", "class"]

for column in required_columns:
    if column not in train_data.columns:
        raise ValueError(
            f"Required column '{column}' is missing from train.csv"
        )

print("\nOriginal dataset shape:", train_data.shape)


# ---------------------------------------
# 3. CHECK CLASS DISTRIBUTION
# ---------------------------------------

print("\nOriginal class distribution:")
print(train_data["class"].value_counts(dropna=False))


# ---------------------------------------
# 4. TEXT CLEANING FUNCTION
# ---------------------------------------

def clean_text(text):

    text = str(text)

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------
# 5. HANDLE MISSING VALUES
# ---------------------------------------

train_data["title"] = train_data["title"].fillna("")
train_data["text"] = train_data["text"].fillna("")

train_data["class"] = (
    train_data["class"]
    .astype(str)
    .str.strip()
    .str.title()
)


# ---------------------------------------
# 6. COMBINE TITLE AND ARTICLE
# ---------------------------------------

train_data["combined_text"] = (
    train_data["title"] + " " + train_data["text"]
)


# ---------------------------------------
# 7. CLEAN COMBINED TEXT
# ---------------------------------------

train_data["combined_text"] = (
    train_data["combined_text"].apply(clean_text)
)


# ---------------------------------------
# 8. REMOVE EMPTY TEXT
# ---------------------------------------

train_data = train_data[
    train_data["combined_text"].str.strip() != ""
].copy()


# ---------------------------------------
# 9. KEEP VALID CLASS LABELS
# ---------------------------------------

train_data = train_data[
    train_data["class"].isin(["Fake", "Real"])
].copy()

if train_data.empty:
    raise ValueError(
        "No valid Fake/Real labels found. Check the class column."
    )

if train_data["class"].nunique() != 2:
    raise ValueError(
        "The dataset must contain both Fake and Real labels."
    )

print("\nCleaned dataset shape:", train_data.shape)

print("\nFinal class distribution:")
print(train_data["class"].value_counts())


# ---------------------------------------
# 10. CHECK DUPLICATE TEXT
# ---------------------------------------

duplicate_count = train_data["combined_text"].duplicated().sum()

print("\nDuplicate text records:", duplicate_count)

conflicting_duplicates = (
    train_data.groupby("combined_text")["class"]
    .nunique()
    .gt(1)
    .sum()
)

print("Texts with conflicting labels:", conflicting_duplicates)


# ---------------------------------------
# 11. SPLIT FEATURES AND LABELS
# ---------------------------------------

X = train_data["combined_text"]

y = train_data["class"]

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_validation))


# ---------------------------------------
# 12. TF-IDF VECTORIZATION
# ---------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_validation_tfidf = vectorizer.transform(X_validation)

print("\nTF-IDF vectorization completed.")

print("Training feature shape:", X_train_tfidf.shape)


# ---------------------------------------
# 13. TRAIN LOGISTIC REGRESSION
# ---------------------------------------

model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    random_state=42
)

print("\nTraining Logistic Regression model...")

model.fit(X_train_tfidf, y_train)

print("\nModel classes:", model.classes_)


# ---------------------------------------
# 14. VALIDATE MODEL
# ---------------------------------------

validation_predictions = model.predict(X_validation_tfidf)

accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

print("\n===================================")
print("       MODEL EVALUATION")
print("===================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ---------------------------------------
# 15. CLASSIFICATION REPORT
# ---------------------------------------

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_validation,
        validation_predictions,
        labels=["Fake", "Real"],
        zero_division=0
    )
)


# ---------------------------------------
# 16. CONFUSION MATRIX
# ---------------------------------------

print("\nCONFUSION MATRIX")

cm = confusion_matrix(
    y_validation,
    validation_predictions,
    labels=["Fake", "Real"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Fake", "Actual Real"],
    columns=["Predicted Fake", "Predicted Real"]
)

print(cm_df)


# ---------------------------------------
# 17. SAVE TRAINED MODEL
# ---------------------------------------

joblib.dump(model, MODEL_FILE)

joblib.dump(vectorizer, VECTORIZER_FILE)

print("\n===================================")
print("       MODEL SAVED SUCCESSFULLY")
print("===================================")

print("\nModel saved at:", MODEL_FILE)

print("Vectorizer saved at:", VECTORIZER_FILE)


# ---------------------------------------
# 18. NEWS PREDICTION FUNCTION
# ---------------------------------------

def predict_news(title, article):

    combined_text = str(title) + " " + str(article)

    cleaned_text = clean_text(combined_text)

    transformed_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(transformed_text)[0]

    probabilities = model.predict_proba(transformed_text)[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    confidence = class_probabilities[prediction] * 100

    return prediction, confidence


# ---------------------------------------
# 19. TEST PREDICTION
# ---------------------------------------

print("\n===================================")
print("       PREDICTION FUNCTION READY")
print("===================================")

print("\nYour Fake News Detection Model is ready.")
