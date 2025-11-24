# Main Actor Gender Prediction Using NLP

## Project Overview
This project aims to predict the **gender of the main actor** in a movie (the first-listed actor) based solely on **user-generated movie reviews** using Natural Language Processing (NLP) and Machine Learning techniques.

It addresses a **binary classification problem**, while handling challenges such as **class imbalance**, as there are significantly more male leads than female leads in the dataset.

The main objective is to demonstrate how textual data from reviews can reveal information about the primary subject of the movie.

## Dataset
The dataset (`movie_reviews.csv`) contains movie reviews with metadata. Below are the key columns:

| Column | Description |
|--------|-------------|
| `Unnamed: 0` | Original index of the row |
| `review_id` | Unique identifier of the review |
| `movie_name` | Name of the movie |
| `year` | Year of the movie release |
| `reviewer_name` | Name of the reviewer |
| `review_text` | Full text of the review (used as input) |
| `rated` | Movie rating |
| `year_api` | Year retrieved from API |
| `genre` | Movie genre(s) |
| `directors` | List of directors |
| `writers` | List of writers |
| `actors` | List of actors |
| `plot` | Plot summary |
| `first_genre` | Primary genre |
| `first_actor` | Name of the first-listed actor |
| `first_director` | Name of the first-listed director |
| `first_writer` | Name of the first-listed writer |
| `first_actor_gender` | Gender of the first actor (**target variable**) |
| `first_director_gender` | Gender of the first director |
| `first_writer_gender` | Gender of the first writer |

**Input feature:** `review_text` — the textual content of the review.  
**Target variable:** `first_actor_gender` — Male / Female

## Tech Stack
- Python 3.12.11 
- Data Manipulation: Pandas, NumPy  
- Visualization: Matplotlib, Seaborn  
- NLP & Machine Learning: Scikit-learn, Spacy  

## Workflow
1. **Data Preprocessing**
    - Cleaning text (removing HTML tags, special characters, multiple spaces)  
    - Removing stopwords but keeping words useful for gender prediction (e.g., "he", "she")  
    - Handling missing values  

2. **Exploratory Data Analysis (EDA)**
    - Distribution of review lengths (words and characters)  
    - Visualizing class imbalance (Male vs. Female)  
    - Identifying the most common words in reviews  

3. **Feature Engineering**
    - Converting text into numerical features using **TF-IDF Vectorizer**  

4. **Model Building & Evaluation**
    - **Multinomial Naive Bayes**: baseline model, good overall accuracy but poor recall for female leads  
    - **Logistic Regression**: implemented with `class_weight='balanced'` to address class imbalance; significantly improved recall  
    - **Linear SVC (Support Vector Classifier)**: hyperparameter tuning using `GridSearchCV`; achieved the best overall balance  

## 📊 Model Performance Comparison

The table below summarizes the performance of the three models tested. Note the significant improvement in handling the minority class ("Female") in Models 2 and 3 compared to the baseline.

| Model | Accuracy | Macro Avg F1 | Female Precision | Female Recall | Male Precision | Male Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1** (MultinomialNB) | 89.49% | 0.83 | **99.20%** | 57.38% | 87.90% | **99.85%** |
| **Model 2** (Logistic Regression) | 93.40% | 0.91 | 84.28% | **89.66%** | **96.60%** | 94.61% |
| **Model 3** (LinearSVC) | **95.33%** | **0.93** | 93.72% | 86.64% | 95.80% | 98.13% |

**Key Observations:**
* **Model 1** had high precision for females but very low recall, missing nearly half of the female leads.
* **Model 2** fixed the recall issue (jumping to ~90%) but sacrificed some precision.
* **Model 3** offered the best balance, achieving the highest overall accuracy and F1-score.

**Conclusion**:

This project successfully demonstrated that natural language processing can predict a main actor's gender based solely on movie review text. A major challenge was the dataset's severe class imbalance, which made standard accuracy scores misleading and necessitated the use of Recall and F1-score for evaluation. 

Initial analysis revealed a reliance on gendered pronouns, so we applied custom stop word filtering to force the models to learn deeper stylistic patterns rather than simply counting words like "he" or "she." While the baseline Multinomial Naive Bayes model showed significant bias against the minority class, implementing a weighted Logistic Regression drastically improved Female Recall to approximately 90%. Ultimately, the Linear Support Vector Classifier (LinearSVC) proved to be the superior model, achieving the highest overall accuracy of 95.3%. This model offered the best trade-off between precision and recall, effectively distinguishing between genders without excessive false positives. 

These results highlight that handling class imbalance and selecting appropriate algorithms are just as critical as feature engineering in text classification tasks. Future iterations could explore sentiment analysis to investigate potential bias in film criticism or employ deep learning models like BERT for capturing contextual nuances. 

## How to Run
1. Clone the repository:
```bash
git clone <repository_url>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter Notebook:
```bash
jupyter notebook "Model predicting gender.ipynb"
```
