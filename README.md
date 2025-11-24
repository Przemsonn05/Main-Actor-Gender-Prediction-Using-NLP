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

## Results

Due to the dataset being heavily imbalanced toward male actors, standard accuracy is insufficient to evaluate performance. We focused on **Precision, Recall, and F1-score**, with a special emphasis on the minority class (Female).

| Model | Accuracy | Female Recall | Female Precision | Female F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial NB** | 87.2% | 47.7% | **99.3%** | 0.64 |
| **Logistic Regression** (Weighted) | 93.4% | **89.7%** | 84.3% | 0.87 |
| **Linear SVC** (Best Model) | **95.3%** | 86.6% | 93.7% | **0.90** |

### Interpretation & Conclusion

The analysis revealed distinct behaviors among the models:

* **Multinomial Naive Bayes (Baseline):** This model exhibited a strong bias towards the majority class. While it achieved decent accuracy, it failed to identify more than half of the female leads (**Recall ~48%**), making it ineffective for this specific problem.
* **Logistic Regression:** Applying `class_weight='balanced'` successfully mitigated the bias, drastically improving the detection of female actors (**Recall ~90%**). However, this came with a trade-off: the model generated more false positives (lower precision).
* **Linear SVC (Winner):** The Support Vector Classifier proved to be the most robust solution. It achieved the **highest overall accuracy (95.3%)** and offered the best balance between sensitivity and correctness. It maintained a high recall for females (86.6%) while offering significantly better precision (93.7%) than Logistic Regression, resulting in the highest **F1-Score (0.90)**.

**Conclusion**:

This project demonstrates how text descriptions can be used to infer actor gender and highlights the importance of using recall and F1-score when dealing with imbalanced datasets.
Proper preprocessing and careful model selection significantly improved the ability to detect the underrepresented class.

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
