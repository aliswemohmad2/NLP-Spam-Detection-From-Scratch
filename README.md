# 📧 NLP Spam Detection - Custom Naive Bayes (From Scratch)

## 📌 Overview
This project implements a Machine Learning binary classification model to detect Spam messages. To demonstrate a deep understanding of core ML algorithms, the **Multinomial Naive Bayes algorithm was built entirely from scratch** using NumPy, bypassing standard out-of-the-box training models like Scikit-Learn's `MultinomialNB`.

## ⚙️ Tech Stack
* **Python 3.x**
* **NumPy:** For high-performance matrix multiplications and log-space calculations.
* **Pandas:** For data manipulation and mapping.
* **Scikit-Learn:** Used *strictly* for text vectorization (`CountVectorizer`) and evaluation metrics, **not** for the algorithm itself.
* **NLTK:** For stop-words removal.
* **Matplotlib:** For visualizing class distributions and F1-Scores.

## 🚀 Key Features & Engineering Highlights
1. **Algorithm From Scratch:** Built a `CustomNaiveBayes` class implementing Prior Probabilities, Conditional Probabilities, and Log-Space calculations to prevent numerical underflow.
2. **Laplace Smoothing:** Handled zero-frequency problems for unseen words by implementing an adjustable Alpha hyperparameter.
3. **Imbalanced Data Handling:** Applied Random Over-sampling to balance the minority class (Spam) with the majority class (Ham), significantly improving the model's Recall.
4. **Grid Search & 5-Fold Cross Validation:** Implemented a custom CV system to find the optimal smoothing parameter (Alpha) without test-set leakage.
5. **Model Explainability:** Added a feature to extract the top 10 most influential words per class, proving the model learned logical patterns (e.g., identifying "free", "win", "prize" as Spam indicators).

## 📊 Results & Performance
After hyperparameter tuning (Best Alpha = 0.1) and data balancing, the model achieved outstanding performance on the unseen Test Set:
* **Overall Accuracy:** 98%
* **Ham Precision & Recall:** ~99%
* **Spam Precision:** 86% | **Spam Recall:** 92% (Optimized via Oversampling)

## 💻 How to Run
1. Clone this repository.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn nltk matplotlib`.
3. Open the Jupyter Notebook (`.ipynb` file) and run the cells sequentially.

---
*Developed as part of a Machine Learning academic project.*
