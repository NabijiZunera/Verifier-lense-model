
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

MODEL_PATH = os.path.join(PROJECT_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_DIR, "tfidf_vectorizer.pkl")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "verifier-lens-development-key"
)


# ==========================================
# 2. LOAD TRAINED MODEL AND VECTORIZER
# ==========================================

model = None
vectorizer = None

try:
    if os.path.isfile(MODEL_PATH) and os.path.isfile(VECTORIZER_PATH):

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        print("SUCCESS: Fake News Model Loaded.")
        print("SUCCESS: TF-IDF Vectorizer Loaded.")
        print("Model Classes:", model.classes_)

    else:
        print("ERROR: Model files not found.")
        print("Model Path:", MODEL_PATH)
        print("Vectorizer Path:", VECTORIZER_PATH)

except Exception as error:
    print("ERROR loading trained model:", error)
    model = None
    vectorizer = None


# ==========================================
# 3. NEWS PREDICTION FUNCTION
# ==========================================

def predict_news(news_text):

    if model is None or vectorizer is None:
        return {
            "class": "unavailable",
            "icon": "⚠️",
            "label": "Model Unavailable",
            "message": "The trained model could not be loaded. Please check the model files.",
            "confidence": 0,
            "text": news_text
        }

    try:
        # Convert input text into TF-IDF features
        transformed_text = vectorizer.transform([news_text])

        # Predict news category
        prediction = model.predict(transformed_text)[0]

        # Normalize predicted label
        predicted_label = str(prediction).strip().lower()

        # REAL NEWS
        if predicted_label in ("real", "1", "true"):

            label = "REAL NEWS"
            result_class = "real"
            icon = "✅"

            message = (
                "The model classified this article as real news. "
                "This prediction is based on learned dataset patterns "
                "and is not independent factual verification."
            )

        # FAKE NEWS
        elif predicted_label in ("fake", "0", "false"):

            label = "FAKE NEWS"
            result_class = "fake"
            icon = "⚠️"

            message = (
                "The model classified this article as fake news. "
                "Please verify its claims using reliable sources."
            )

        else:
            return {
                "class": "unavailable",
                "icon": "⚠️",
                "label": "Unknown Model Label",
                "message": f"The model returned an unexpected label: {prediction}",
                "confidence": 0,
                "text": news_text
            }

        # Calculate prediction probability
        confidence = 0

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(transformed_text)[0]

            confidence = round(
                float(max(probabilities)) * 100,
                2
            )

        return {
            "class": result_class,
            "icon": icon,
            "label": label,
            "message": message,
            "confidence": confidence,
            "text": news_text
        }

    except Exception as error:

        print("PREDICTION ERROR:", error)

        return {
            "class": "unavailable",
            "icon": "⚠️",
            "label": "Prediction Error",
            "message": "Unable to analyze this article. Please try again.",
            "confidence": 0,
            "text": news_text
        }


# ==========================================
# 4. HOME / DASHBOARD PAGE
# ==========================================

@app.route("/")
@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        username="User"
    )


# ==========================================
# 5. NEWS DETECTION PAGE
# ==========================================

@app.route("/news", methods=["GET", "POST"])
def news():

    if request.method == "GET":

        return render_template(
            "news.html",
            result=None
        )

    # Get submitted news article
    news_text = request.form.get(
        "news_text",
        ""
    ).strip()

    # Validate article
    if len(news_text) < 10:

        result = {
            "class": "unavailable",
            "icon": "ℹ️",
            "label": "Article Too Short",
            "message": "Please enter at least 10 characters.",
            "confidence": 0,
            "text": news_text
        }

        return render_template(
            "news.html",
            result=result
        )

    # Generate prediction
    result = predict_news(news_text)

    return render_template(
        "news.html",
        result=result
    )


# ==========================================
# 6. RESOURCES PAGE
# ==========================================

@app.route("/resources")
def resources():

    return render_template(
        "dashboard.html",
        username="User"
    )


# ==========================================
# 7. HISTORY PAGE
# ==========================================

@app.route("/history")
def history():

    return render_template(
        "dashboard.html",
        username="User"
    )


# ==========================================
# 8. TEAM PAGE
# ==========================================

@app.route("/team")
def team():

    return redirect(
        url_for("dashboard") + "#team-section"
    )


# ==========================================
# 9. LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    return redirect(
        url_for("dashboard")
    )


# ==========================================
# 10. APPLICATION HEALTH CHECK
# ==========================================

@app.route("/health")
def health():

    if model is not None and vectorizer is not None:

        return {
            "status": "healthy",
            "model_loaded": True,
            "vectorizer_loaded": True
        }, 200

    return {
        "status": "error",
        "model_loaded": False,
        "vectorizer_loaded": False
    }, 503


# ==========================================
# 11. RUN FLASK APPLICATION LOCALLY
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )



   