# NLP Review Analysis and Prediction Application

## Project Description

This project showcases a complete end-to-end Natural Language Processing (NLP) pipeline, integrating both supervised and unsupervised learning techniques for analyzing customer reviews.  
The final deliverable is an **interactive Streamlit application** that provides functionalities such as sentiment analysis, star rating prediction, topic modeling, and review summarization using state-of-the-art machine learning and NLP techniques.

---

## Features

### 1. **Data Cleaning and Preprocessing**
- Text cleaning and standardization:
  - Removal of punctuation and special characters.
  - Lowercasing and tokenization.
  - Spelling correction using `TextBlob` and custom rules.
- Translating non-English reviews (e.g., French to English) using `TextBlob`.
- Creation of additional features:
  - Review length.
  - Sentiment polarity score.
- Dataset preparation for supervised tasks:
  - Star rating classification (1–5 stars).
  - Sentiment classification (positive, neutral, negative).

### 2. **Supervised Learning**
- **Star Rating Prediction**:
  - Predicts star ratings (1–5) using a **Random Forest Classifier** trained on `TF-IDF` features.
- **Sentiment Analysis**:
  - Classifies reviews as positive, neutral, or negative using a Random Forest model.

### 3. **Unsupervised Learning**
- **Topic Modeling**:
  - Implements Latent Dirichlet Allocation (LDA) for extracting underlying topics.
  - Visualizes topic distributions with `pyLDAvis`.

### 4. **Embedding and Semantic Similarity**
- Trains **Word2Vec** embeddings for semantic understanding of words.  
- Computes semantic similarity using cosine similarity for meaningful insights.

### 5. **Explainable AI**
- **SHAP**:
  - Provides explainability for predictions with SHAP values.
  - Visualizes features contributing to star rating predictions.

### 6. **Interactive Streamlit Application**
- **Core Functionalities**:
  - **Star Rating Prediction**: Predict the number of stars (1–5) for a given review.  
  - **Sentiment Analysis**: Determine whether the sentiment is positive, neutral, or negative.  
  - **Review Summarization**: Summarize input reviews using pre-trained models.  
  - **Prediction Explainability**: Display key features influencing star rating predictions.  
  - **RAG-Based Summaries**: Retrieve and summarize reviews similar to the input.  
  - **Question Answering**: Answer questions about the dataset using an NLP QA pipeline.  
- **Visualizations**:
  - Topic distributions, SHAP explanations, and semantic similarities.

---

## 🚀 Installation and Usage

### Prerequisites
- Python 3.10 or higher
- Install required libraries using `pip`.

1. Clone the repository:

```bash
git clone https://github.com/YahyaELOUDOUNI/NLP-Review-Analysis-and-Prediction-Application.git
cd NLP-Review-Analysis-Application
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to explore the application.

---

## 📁 Project Directory Structure

```
├── deep_learning_model/
│   ├── assets/            # TensorFlow metadata
│   ├── variables/         # Model weights and configurations
│   ├── saved_model.pb     # Saved TensorFlow model
│   └── tf_model.h5        # Optional deep learning model in H5 format
├── app.py                 # Main Streamlit app script
├── cleaned_reviews.csv    # Cleaned and preprocessed dataset
├── Data_cleaning_and_analysis.ipynb  # Data preparation and exploration
├── random_forest_model.pkl  # Random Forest model for star ratings
├── rf_sentiment_model.pkl   # Random Forest model for sentiment analysis
├── tfidf_vectorizer.pkl     # TF-IDF vectorizer for rating predictions
├── tfidf_vectorizer_sentiment.pkl  # TF-IDF vectorizer for sentiment analysis
├── README.md               # Project documentation
```

---

## 📷 Application Interface

### Key Functionalities
#### Rating Prediction
Predict the star rating based on the content of a review.

#### Sentiment Analysis
Determine if a review has positive, neutral, or negative sentiment.

#### Review Summarization
Generate concise summaries of reviews.

#### Prediction Explanation
Visualize SHAP explanations for model predictions.

#### Semantic Similarity
Retrieve top 3 reviews similar to the input text.

#### Question Answering
Answer questions based on the dataset using an NLP pipeline.

---

## 🛠 Technologies and Libraries Used

### Programming Language:
- Python 3.10+

### Web Framework:
- Streamlit: Interactive application development.

### NLP Libraries:
- TextBlob: Translation, preprocessing, and sentiment analysis.  
- Transformers: Hugging Face library for summarization and question answering.  
- TensorFlow Hub: Universal Sentence Encoder (USE) embeddings.  
- Gensim: Topic modeling and LDA visualization.  
- NLTK: Tokenization, stopword removal, and normalization.  
- LanguageTool: Advanced spelling correction.  

### Machine Learning Libraries:
- Scikit-Learn: Random Forest models and TF-IDF vectorizer.  
- SHAP: Explainable AI for visualizing feature contributions.  

### Deep Learning Framework:
- TensorFlow/Keras: Deep learning models for star rating predictions.  
- Universal Sentence Encoder (USE): Semantic similarity embeddings.  

### Visualization Libraries:
- Matplotlib, pyLDAvis, and Seaborn.

### Utilities:
- NumPy, Pandas, and Regular Expressions (`re`).

---

### 🚀 Questions?

Feel free to connect with me on LinkedIn: [Yahya EL OUDOUNI](https://www.linkedin.com/in/yahya-el-oudouni/)