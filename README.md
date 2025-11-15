# 1. AI Powered Email Spam Detection
**AI-Powered Spam Detection with Naïve Bayes & Bag-of-Words**
> Classify emails as **spam** or **ham** with **98.7% accuracy** using clean, modular ML pipelines.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.9%2B-6A5ACD?logo=nltk&logoColor=white)](https://www.nltk.org)

---

# 2. Objective
Develop a **robust, maintainable, and deployable** email classification system with:
- End-to-end **reproducible preprocessing**
- **High generalization** on unseen data
- **Zero-dependency inference** via serialized model

# 3. Tech Stack
| Component | Version | Role |
|---------|--------|------|
| **Python** | `3.10+` | Core runtime |
| **Pandas** | `2.2+` | Data loading & preprocessing |
| **NLTK** | `3.9+` | Lemmatization, stopwords |
| **Scikit-learn** | `1.4+` | `CountVectorizer`, `MultinomialNB`, pipeline |
| **Pickle** | Built-in | Model serialization |
| **Regex (`re`)** | Built-in | Pattern-based cleaning |

---

# 4. Key Features
| Feature | Benefit |
|--------|--------|
| **Advanced Text Cleaning** | Strip HTML, URLs, emails, numbers, special chars |
| **Smart Normalization** | Lowercase + lemmatization + stopword removal |
| **Efficient BoW** | 3,500 most frequent tokens → sparse matrix |
| **Multinomial Naïve Bayes** | Proven for text; fast training & inference |
| **Model + Vectorizer Persistence** | `pickle` → instant deployment |
| **Modular Architecture** | `src/` → reusable, testable, scalable |

---

# 5. Quick Start

## 5.1. Clone & Install
```bash
git clone https://github.com/Rsivamani/spam-email-classifier.git
cd spam-email-classifier
```
---

# 6. Multinomial Naïve Bayes with Bag-of-Words

## 6.1. Model Details
- **Model Type**: Multinomial Naïve Bayes + Bag-of-Words (3500 features)
- **Dataset Split**: 80% train / 20% test

## 6.2. Preprocessing
- Lemmatization
- Stopword removal
- Regex cleaning

---

# 7. Model Performance Metrics

##  7.1 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** | 0.98 | 0.99 | 0.99 | 1,450 |
| **1** | 0.93 | 0.90 | 0.91 | 222 |

## 7.2 Overall Metrics
- **Accuracy**: 0.98 (1,672 samples)
- **Macro Average**: 0.96 precision, 0.94 recall, 0.95 f1-score
- **Weighted Average**: 0.98 precision, 0.98 recall, 0.98 f1-score

---

# 8. Conclusion

- The AI-Powered Spam Email Classifier delivers a reliable, modular, and production-focused solution for identifying spam and non-spam emails.  
-  Its architecture combines rigorous text preprocessing, a scalable Bag-of-Words feature representation, and a high-performance Multinomial Naïve Bayes model to achieve an accuracy of 98% on real-world data.

 
