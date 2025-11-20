# Main Actor Gender Prediction Using NLP

# Main Character Gender Prediction from Movie Reviews

## 📌 Project Overview
This project aims to predict the gender of a movie's main character based solely on the textual content of user-generated movie reviews. using Natural Language Processing (NLP) and Machine Learning techniques.

The project addresses a classic binary classification problem while tackling challenges such as **class imbalance** (significantly more male leads than female leads in the dataset).

## 📂 Dataset
The dataset (`movie_reviews.csv`) consists of movie reviews alongside metadata.
* **Input:** `review_text` (The content of the review).
* **Target:** `first_actor_gender` (Male/Female).

## 🛠️ Tech Stack
* **Python 3.x**
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **NLP & Machine Learning:** Scikit-learn, Spacy

## 📊 Workflow

1.  **Data Preprocessing:**
    * Cleaning text (removing HTML tags, special characters).
    * Stopwords removal.
    * Handling missing values.
2.  **Exploratory Data Analysis (EDA):**
    * Analysis of review length distribution.
    * Visualizing class imbalance (Male vs. Female leads).
    * Identifying most common words used in reviews.
3.  **Feature Engineering:**
    * Converting text data into numerical format using **TF-IDF Vectorizer**.
4.  **Model Building & Evaluation:**
    * **Multinomial Naive Bayes:** Baseline model. High accuracy but poor recall for the minority class.
    * **Logistic Regression:** Implemented with `class_weight='balanced'` to handle dataset imbalance. Significant improvement in Recall.
    * **Linear SVC (Support Vector Classifier):** Hyperparameter tuning using `GridSearchCV`.

## 📈 Results

The dataset was heavily imbalanced towards male characters. Standard accuracy was misleading, so precision, recall, and F1-score were crucial metrics.

| Model | Accuracy | Female Class Recall | Male Class Recall |
| :--- | :---: | :---: | :---: |
| **Naive Bayes** | 87% | ~48% | ~99% |
| **Logistic Regression** | 93% | ~90% | ~94% |
| **Linear SVC (Best)** | **95%** | **~86%** | **~98%** |

**Conclusion:** The Linear SVC model proved to be the most effective, offering the best balance between Precision and Recall for both genders.

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/gender-prediction-nlp.git](https://github.com/YOUR_USERNAME/gender-prediction-nlp.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Model predicting gender.ipynb"
    ```

## 👤 Author
[Your Name/LinkedIn Profile]
