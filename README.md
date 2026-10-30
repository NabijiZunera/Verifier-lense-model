# Verifier Lens – Fake News Detection System

Fake News Detection System using Machine Learning and Natural Language Processing (NLP).

## Abstract

### 1. Background
The rapid growth of digital media and online news platforms has increased the spread of misinformation and fabricated news. Fake news can influence public opinion, create confusion, and reduce trust in reliable information sources. Therefore, an automated system is required to assist users in identifying potentially fake and real news content.

### 2. Objective
The main objective of **Verifier Lens – Fake News Detection System** is to develop a machine-learning-based web application that automatically classifies news text into two categories: REAL and FAKE. The system aims to provide users with an accessible interface for submitting news content and receiving prediction results.

### 3. Methodology
The proposed system uses Natural Language Processing (NLP) and supervised machine-learning techniques. A labelled real-and-fake news dataset is used for model training. Text data is processed and converted into numerical features using Term Frequency–Inverse Document Frequency (TF-IDF). Logistic Regression is then applied to train the classification model to distinguish between real and fake news articles.

### 4. System Implementation
The trained machine-learning model and TF-IDF vectorizer are saved using Python's model-persistence functionality. Flask is used as the web framework to connect the trained model with the user interface. Users can enter news content through the web application, which processes the submitted text and displays the predicted category along with a confidence score when supported by the model.

### 5. Results and Discussion
The developed application demonstrates the integration of NLP, machine learning, and web development for automated news-text classification. It generates REAL or FAKE predictions for submitted text. The actual classification accuracy and other evaluation metrics should be reported after testing the trained model on an appropriate labelled test dataset.

### 6. Conclusion
Verifier Lens provides a practical approach to identifying potentially misleading news content using machine-learning techniques. The project demonstrates how TF-IDF and Logistic Regression can be integrated into a Flask-based application. Future improvements may include source verification, URL-based news analysis, advanced language models, and additional evidence-based fact-checking features.

**Keywords:** Fake News Detection, Verifier Lens, Natural Language Processing, Machine Learning, TF-IDF, Logistic Regression, Text Classification, Flask, News Verification.

