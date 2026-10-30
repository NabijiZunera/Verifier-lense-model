
from flask import Flask, render_template, request, redirect, url_for
import os
import joblib

# ==========================================
# 1. PROJECT PATHS AND FLASK CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

TEMPLATE_DIR = os.path.join(PROJECT_DIR, "templates")
STATIC_DIR = os.path.join(PROJECT_DIR, "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

app.secret_key = "verifier-lens-secret-key"


# ==========================================
# 2. LOAD TRAINED MODEL AND VECTORIZER
# ==========================================

MODEL_PATH = os.path.join(PROJECT_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_DIR, "tfidf_vectorizer.pkl")

model = None
vectorizer = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        print("SUCCESS: Fake news model loaded.")
        print("SUCCESS: TF-IDF vectorizer loaded.")

    else:
        print("WARNING: Model files were not found.")
        print("Expected model:", MODEL_PATH)
        print("Expected vectorizer:", VECTORIZER_PATH)
        print("News prediction is currently unavailable.")

except Exception as e:
    print("ERROR loading the model:", e)
    model = None
    vectorizer = None


# ==========================================
# 3. HELPER FUNCTION FOR PREDICTION
# ==========================================

def predict_news(news_text):
    """
    Predict whether the supplied news text is REAL or FAKE.

    Assumes the trained model uses:
        0 = FAKE
        1 = REAL

    Verify these labels against your training code.
    """

    if model is None or vectorizer is None:
        return {
            "class": "unavailable",
            "icon": "⚠️",
            "label": "Model Unavailable",
            "message": (
                "The trained model could not be loaded. "
                "Check that the model and vectorizer files exist."
            ),
            "confidence": 0,
            "text": news_text
        }

    try:
        # Convert news text into TF-IDF features
        transformed_text = vectorizer.transform([news_text])

        # Predict the class
        prediction = model.predict(transformed_text)[0]

        # Convert the predicted class into a readable label.
        # These mappings must match your model's training labels.
        if str(prediction).strip().lower() in ("1", "real", "true"):
            label = "REAL NEWS"
            result_class = "real"
            icon = "✅"
            message = (
                "The model classified this article as real news. "
                "This is a model prediction, not independent verification."
            )

        elif str(prediction).strip().lower() in ("0", "fake", "false"):
            label = "FAKE NEWS"
            result_class = "fake"
            icon = "⚠️"
            message = (
                "The model classified this article as fake news. "
                "Verify the claims using reliable sources."
            )

        else:
            return {
                "class": "unavailable",
                "icon": "⚠️",
                "label": "Unknown Model Label",
                "message": (
                    "The model returned an unfamiliar class label. "
                    "Check the labels used during training."
                ),
                "confidence": 0,
                "text": news_text
            }

        # Calculate confidence when the model supports it
        confidence = 0

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed_text)[0]
            confidence = round(float(max(probabilities)) * 100, 2)

        return {
            "class": result_class,
            "icon": icon,
            "label": label,
            "message": message,
            "confidence": confidence,
            "text": news_text
        }

    except Exception as e:
        print("Prediction error:", e)

        return {
            "class": "unavailable",
            "icon": "⚠️",
            "label": "Prediction Error",
            "message": (
                "The model could not analyze this article. "
                "Check the terminal for the error details."
            ),
            "confidence": 0,
            "text": news_text
        }


# ==========================================
# 4. HOME / DASHBOARD PAGE
# ==========================================

@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", username="User")


# ==========================================
# 5. NEWS DETECTION PAGE
# ==========================================

@app.route("/news", methods=["GET", "POST"])
def news():

    # Display the page when opened normally
    if request.method == "GET":
        return render_template("news.html", result=None)

    # Receive the submitted article
    news_text = request.form.get("news_text", "").strip()

    # Validate the input
    if len(news_text) < 10:
        return render_template(
            "news.html",
            result={
                "class": "unavailable",
                "icon": "ℹ️",
                "label": "Article Too Short",
                "message": "Please enter at least 10 characters.",
                "confidence": 0,
                "text": news_text
            }
        )

    # Run prediction
    result = predict_news(news_text)

    # Display prediction on the page
    return render_template("news.html", result=result)


# ==========================================
# 6. RESOURCES PAGE
# ==========================================

@app.route("/resources")
def resources():
    return render_template("dashboard.html", username="User")


# ==========================================
# 7. HISTORY PAGE
# ==========================================

@app.route("/history")
def history():
    return render_template("dashboard.html", username="User")


# ==========================================
# 8. TEAM / DASHBOARD SECTION
# ==========================================

@app.route("/team")
def team():
    return redirect(url_for("dashboard") + "#team-section")


# ==========================================
# 9. LOGOUT ROUTE
# ==========================================

@app.route("/logout")
def logout():
    return redirect(url_for("dashboard"))


# ==========================================
# 10. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)


   