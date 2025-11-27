# Main Actor Gender Prediction Using NLP

## 📋 Project Overview

This project predicts the **gender of the main actor** in a movie (the first-listed actor) based solely on **user-generated movie reviews** using Natural Language Processing (NLP) and Machine Learning techniques.

It tackles a **binary classification problem** while addressing significant challenges such as **class imbalance**, where the dataset contains substantially more male leads than female leads. The project demonstrates how textual data from reviews can reveal meaningful information about the primary subject of a film.

### Key Features
- **3 Machine Learning Models** compared: Multinomial Naive Bayes, Logistic Regression, and Linear SVC
- **Best Model Performance**: 95.33% accuracy with balanced precision and recall across both genders
- **Class Imbalance Handling**: Techniques including weighted class balancing and hyperparameter optimization
- **Comprehensive EDA**: Visualizations of review distributions, word frequencies, and dataset composition
- **Production-Ready**: Trained model saved and ready for inference

---

## 📊 Dataset

The dataset (`movie_reviews.csv`) contains 16,500+ movie reviews with comprehensive metadata. 

### Key Columns
| Column | Description |
|--------|-------------|
| `review_id` | Unique identifier for each review |
| `movie_name` | Title of the movie |
| `review_text` | Full text of the review (primary input feature) |
| `first_actor_gender` | Gender of the first-listed actor (target variable: Male/Female) |
| `year_of_production` | Year the movie was released |
| `year_of_opinion` | Year the review was posted |
| `rated` | Movie rating (G, PG, R, etc.) |
| `first_genre` | Primary genre category |
| `first_actor` | Name of the main actor |
| `first_director` | Name of the main director |

### Data Characteristics
- **Class Distribution**: ~12,000 male leads vs. ~4,500 female leads (2.7:1 imbalance)
- **Review Length**: Median ~600-800 words, with right-skewed distribution
- **Challenge**: Significant class imbalance requires careful model evaluation and weighting strategies

---

## 🔄 Project Workflow

### 1. **Data Preprocessing**
- Text cleaning: removing HTML tags, special characters, and extra whitespace
- Custom stopword filtering to preserve gendered pronouns ("he", "she") for meaningful feature learning
- Handling missing values by filling with "unidentified" placeholders
- Feature engineering: calculating review word and character counts

### 2. **Exploratory Data Analysis (EDA)**
- Distribution analysis of review lengths (words and characters)
- Class imbalance visualization (Male vs. Female ratio)
- Most frequently used words in reviews
- Movie and director popularity analysis
- Gender distribution across top films and directors

### 3. **Feature Engineering**
- **TF-IDF Vectorization**: Converting text into numerical features with term frequency-inverse document frequency weighting
- **Train-Test Split**: 80-20 split with random_state=101 for reproducibility

### 4. **Model Development & Evaluation**
Three models were compared with hyperparameter tuning via GridSearchCV:

#### **Model 1: Multinomial Naive Bayes (Baseline)**
- **Pros**: Fast training, simple baseline
- **Cons**: Poor recall for minority class (57% for females)
- **Use Case**: Establishes performance baseline

#### **Model 2: Logistic Regression (with Class Weighting)**
- **Key Feature**: `class_weight='balanced'` to address imbalance
- **Pros**: Significant improvement in female recall (~90%)
- **Cons**: Slightly lower precision for female class (~84%)
- **Use Case**: Balanced approach when catching minorities is important

#### **Model 3: Linear SVC (Recommended)**
- **Hyperparameters Tuned**: C, penalty (L2), loss function (hinge/squared_hinge)
- **Pros**: Best overall balance, highest accuracy (95.33%)
- **Cons**: Slightly lower recall for females compared to Logistic Regression
- **Use Case**: Production deployment for reliable gender classification

---

## 📈 Model Performance Comparison

| Model | Accuracy | F1-Score | Female Precision | Female Recall | Male Precision | Male Recall |
|:------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Bayes** | 89.49% | 0.83 | 99.20% | 57.38% | 87.90% | 99.85% |
| **Logistic Regression** | 93.40% | 0.91 | 84.28% | 89.66% | 96.60% | 94.61% |
| **Linear SVC** ⭐ | **95.33%** | **0.93** | 93.72% | 86.64% | 95.80% | 98.13% |

### Key Insights

**Model 1 Limitations**:
- Extremely imbalanced performance: 99.85% male recall vs. 57.38% female recall
- Biased toward majority class due to class imbalance
- Not suitable for production where both classes matter equally

**Model 2 Breakthrough**:
- Class weighting dramatically improved female detection (~90% recall)
- Achieved balanced performance across genders (F1: 0.91)
- Trade-off: slight precision decrease acceptable for practical use

**Model 3 Victory**:
- Best overall metrics with 95.33% accuracy
- Highest precision for female class (93.72%)
- Maintains strong recall for both classes
- Optimal choice for production deployment

---

## 🎯 Key Findings

1. **Linguistic Patterns Exist**: Movie reviews contain subtle linguistic cues about the main character's gender beyond simple pronoun usage
2. **Class Imbalance is Critical**: Standard accuracy is misleading; F1-score and recall are essential metrics
3. **Hyperparameter Tuning Matters**: GridSearchCV optimization improved LinearSVC performance significantly
4. **Algorithm Selection is Crucial**: LinearSVC outperformed traditional approaches for high-dimensional text data
5. **Bias in Data**: Dataset reflects real-world gender imbalance in film industry (more male leads reviewed)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.12.11 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML/NLP** | Scikit-learn, TF-IDF Vectorizer |
| **Model Persistence** | Joblib |
| **Notebook** | Jupyter |

---

## 🚀 How to Run

### Installation & Execution

1. **Clone the repository**:
   ```bash
   git clone <repository url>
   ```

3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook "notebooks/Model predicting gender (1).ipynb"
   ```

---