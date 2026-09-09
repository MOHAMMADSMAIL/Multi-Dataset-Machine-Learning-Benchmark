# Multi-Dataset Machine Learning Benchmark

A notebook-based benchmark that downloads user-provided CSV datasets from Google Drive, applies a shared preprocessing pipeline, trains **50 scikit-learn algorithms** across four machine-learning categories, evaluates the successful runs, and produces comparative plots for the top and bottom performers.

> This README documents the supplied notebook and its recorded outputs only. No unrecorded results or dataset provenance has been added.

## Overview

The project is designed to compare a broad collection of supervised and unsupervised algorithms on multiple tabular datasets. For each uploaded dataset, the notebook performs data cleaning, identifies the target column, creates a reproducible train/test split, scales the features, trains the configured models, saves the result records to CSV, and visualizes category-level comparisons.

The supplied run processed three datasets:

| Dataset | Rows | Columns | Target used by the run | Input mechanism |
|---|---:|---:|---|---|
| `Dataset_1_ID_1oQnqc` | 253,680 | 22 | `Diabetes_012` | Google Drive CSV |
| `Dataset_2_ID_1YB7Oz` | 70,692 | 22 | `Income` | Google Drive CSV; fallback to last column |
| `Dataset_3_ID_1caYPh` | 253,680 | 22 | `Income` | Google Drive CSV; fallback to last column |

The notebook does not specify the original dataset names, feature descriptions, licensing, or a public source citation beyond the Google Drive links entered in the notebook.

## Key Features

- Accepts one or more Google Drive file links and extracts their file IDs.
- Downloads and loads CSV datasets with `gdown` and `pandas`.
- Handles several missing-value tokens: empty strings, `nan`, `null`, `none`, and `?`.
- Uses mean imputation for numeric-looking columns and mode imputation for other columns.
- Converts the feature matrix and target to numeric values.
- Uses an 80/20 train/test split with `random_state=42`.
- Applies `StandardScaler` to the training and test features.
- Limits model fitting and clustering to a reproducible sample of up to 1,000 training rows.
- Evaluates classification, regression, and clustering runs and records dimensionality-reduction successes.
- Saves results to `ColabData/custom_datasets_results.csv`.
- Produces per-dataset top-five/worst-five plots and overall comparison plots.

## Project Workflow

```text
Google Drive CSV links
        ↓
Download and load datasets
        ↓
Missing-value handling and numeric conversion
        ↓
Target-column selection
        ↓
80/20 train/test split
        ↓
StandardScaler fitted on training data
        ↓
Up to 1,000 sampled training rows for model fitting
        ↓
50 algorithms across four categories
        ↓
Metrics and status records
        ↓
CSV export and comparison plots
```

## Algorithms

The notebook defines exactly **50 algorithms** in four categories.

### Classification — 15 algorithms

1. Logistic Regression
2. Ridge Classifier
3. SGD Classifier
4. Perceptron
5. Passive Aggressive
6. Linear SVC
7. SVC with RBF kernel
8. K-Nearest Neighbors
9. Gaussian Naive Bayes
10. Multinomial Naive Bayes
11. Bernoulli Naive Bayes
12. Decision Tree
13. Random Forest
14. Extra Trees
15. HistGradientBoosting

### Regression — 15 algorithms

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Huber Regressor
6. SGD Regressor
7. Support Vector Regression
8. K-Nearest Neighbors Regressor
9. Decision Tree Regressor
10. Random Forest Regressor
11. Extra Trees Regressor
12. HistGradientBoosting Regressor
13. Gradient Boosting Regressor
14. AdaBoost Regressor
15. Poisson Regressor

### Clustering — 10 algorithms

1. K-Means
2. Mini-Batch K-Means
3. Bisecting K-Means
4. Agglomerative Clustering
5. Birch
6. DBSCAN
7. OPTICS
8. Mean Shift
9. Spectral Clustering
10. Gaussian Mixture

### Dimensionality Reduction — 10 algorithms

1. PCA
2. Incremental PCA
3. Kernel PCA
4. Truncated SVD
5. NMF
6. FastICA
7. Factor Analysis
8. Isomap
9. Locally Linear Embedding
10. MDS

## Evaluation Metrics

The executed evaluation code records the following metrics:

- **Accuracy (%)** and weighted **F1 score** for classification.
- **MAE**, **RMSE**, and **R²** for regression.
- **Silhouette score** and number of detected clusters for clustering.
- Number of transformed components and execution time for dimensionality reduction.

`calinski_harabasz_score` is imported in the notebook but is not used in the executed result records or plots.

## Recorded Results

The following values are taken directly from the notebook’s stored output. “Best” and “worst” follow the notebook’s own ordering: highest accuracy for classification, lowest MAE for regression, and highest/lowest silhouette score for clustering.

### Dataset 1 — `Dataset_1_ID_1oQnqc`

| Category | Best recorded result | Worst recorded result |
|---|---|---|
| Classification | Ridge Classifier — **84.27% accuracy** | Gaussian Naive Bayes — **8.76% accuracy** |
| Regression | Huber Regressor — **MAE 0.3027** | AdaBoost Regressor — **MAE 0.5628** |
| Clustering | Spectral Clustering — **silhouette 0.2759** | OPTICS — **silhouette -0.1553** |

Other recorded classification leaders were SVC RBF (84.24%), Linear SVC (84.20%), Random Forest (84.02%), and Logistic Regression (83.86%). Other recorded regression leaders were Support Vector Regression (0.3664 MAE), K-Nearest Neighbors Regressor (0.4079), Gradient Boosting Regressor (0.4247), and Extra Trees Regressor (0.4317).

### Dataset 2 — `Dataset_2_ID_1YB7Oz`

| Category | Best recorded result | Worst recorded result |
|---|---|---|
| Classification | Ridge Classifier — **32.58% accuracy** | Gaussian Naive Bayes — **12.87% accuracy** |
| Regression | Support Vector Regression — **MAE 1.3749** | Decision Tree Regressor — **MAE 1.8615** |
| Clustering | Spectral Clustering — **silhouette 0.2370** | OPTICS — **silhouette -0.1936** |

Other recorded classification leaders were Linear SVC (32.48%), Logistic Regression (32.12%), SVC RBF (32.01%), and Multinomial Naive Bayes (30.79%). Other recorded regression leaders were Huber Regressor (1.3876 MAE), Gradient Boosting Regressor (1.3896), Linear Regression (1.4010), and Ridge Regression (1.4010).

### Dataset 3 — `Dataset_3_ID_1caYPh`

| Category | Best recorded result | Worst recorded result |
|---|---|---|
| Classification | Ridge Classifier — **37.41% accuracy** | Passive Aggressive — **18.64% accuracy** |
| Regression | Support Vector Regression — **MAE 1.3077** | Decision Tree Regressor — **MAE 1.7177** |
| Clustering | Spectral Clustering — **silhouette 0.2734** | OPTICS — **silhouette -0.1840** |

Other recorded classification leaders were Linear SVC (37.28%), Logistic Regression (36.97%), SVC RBF (36.79%), and Multinomial Naive Bayes (35.68%). Other recorded regression leaders were Huber Regressor (1.3162 MAE), Ridge Regression (1.3306), Linear Regression (1.3306), and SGD Regressor (1.3333).

## Visualizations

All images below were extracted from the notebook’s actual embedded PNG outputs and renamed for clarity. The plots show the recorded run; they are not newly generated benchmark results.

### Dataset 1

![Dataset 1 classification results](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9kYXRhc2V0MV9jbGFzc2lmaWNhdGlvbl9yZXN1bHRz.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTlrWVhSaGMyVjBNVjlqYkdGemMybG1hV05oZEdsdmJsOXlaWE4xYkhSei5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTA4MTI4MDB9fX1dfQ__&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIDefF9qai6LQ-0qfEQXkhOK22L7RL~fBhrKBgbQYR~LBAiEAu5yuBQRZXreSS5STEvpdkM9detTxuhY9XPDZuni1bFQ_)

![Dataset 1 regression results](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9kYXRhc2V0MV9yZWdyZXNzaW9uX3Jlc3VsdHM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTlrWVhSaGMyVjBNVjl5WldkeVpYTnphVzl1WDNKbGMzVnNkSE0ucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzkwODEyODAwfX19XX0_&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIBl-tJ47uywNPjyrHqrvIrCj3mS78h63QAaLYAdiCAlUAiEAtEstqv1oszSQKhbCK5HwlsVD7nQM1eJDoQvNauzFhWE_)

![Dataset 1 clustering results](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9kYXRhc2V0MV9jbHVzdGVyaW5nX3Jlc3VsdHM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTlrWVhSaGMyVjBNVjlqYkhWemRHVnlhVzVuWDNKbGMzVnNkSE0ucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzkwODEyODAwfX19XX0_&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIHmRJakakJjrON8MzpxtT9H9fwvvDs9IcHE5chIqJ65AAiEA8AqB5W64qvdj1~UIxkb3XdurxJ6kJrgJpS7aa-u1FwQ_)

### Dataset 2

![Dataset 2 classification results](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9kYXRhc2V0Ml9jbGFzc2lmaWNhdGlvbl9yZXN1bHRz.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTlrWVhSaGMyVjBNbDlqYkdGemMybG1hV05oZEdsdmJsOXlaWE4xYkhSei5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTA4MTI4MDB9fX1dfQ__&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEQCIACCdrbWN3XzsPrvm2qxhTuWRwhPgKqcX7VJBG-v3vz0AiBl4g9d-l39ShwLJ7LkkQ6pkKqp-bBbaANdAEh-g6X8bQ__)

![Dataset 2 regression results](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9kYXRhc2V0Ml9yZWdyZXNzaW9uX3Jlc3VsdHM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTlrWVhSaGMyVjBNbDl5WldkeVpYTnphVzl1WDNKbGMzVnNkSE0ucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzkwODEyODAwfX19XX0_&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEQCIGqLH1DXYl1akM-u4TvjPgYiAkBIQms9FLhOmKk9ymgsAiAgqCEgbZ8GV6ebjzsBRryJkTqoBS6XR6PU4x6qld7uyA__)

![Dataset 2 clustering results](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9kYXRhc2V0Ml9jbHVzdGVyaW5nX3Jlc3VsdHM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTlrWVhSaGMyVjBNbDlqYkhWemRHVnlhVzVuWDNKbGMzVnNkSE0ucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzkwODEyODAwfX19XX0_&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEYCIQDYLJZckddY6-ssSvo4W5DkLcgAISiqk6tE~Yc7Rgy6DgIhAL6ng2yb0puPiCAkn5AOQe6Lg-ORtUqqbPfAQNx1u0Ls)

### Dataset 3

![Dataset 3 classification results](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9kYXRhc2V0M19jbGFzc2lmaWNhdGlvbl9yZXN1bHRz.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTlrWVhSaGMyVjBNMTlqYkdGemMybG1hV05oZEdsdmJsOXlaWE4xYkhSei5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTA4MTI4MDB9fX1dfQ__&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEYCIQCNKpO2JQnsAdTVLeypbbQLGXIaoUCAxs8oRwzu~MmstwIhALZZA86fQn~q4aXgTFks0UFIVpRvwdB-pvkKEl30CHR8)

![Dataset 3 regression results](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9kYXRhc2V0M19yZWdyZXNzaW9uX3Jlc3VsdHM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTlrWVhSaGMyVjBNMTl5WldkeVpYTnphVzl1WDNKbGMzVnNkSE0ucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzkwODEyODAwfX19XX0_&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIQCAoP6~ncPUUXBKCYMIoV~7gYqndBjmiDyA14Rl53ehaAIgAT2aVRkXGL5HevFwRP7POJmbPG6tFBhrBprPIOjW5Hc_)

![Dataset 3 clustering results](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9kYXRhc2V0M19jbHVzdGVyaW5nX3Jlc3VsdHM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTlrWVhSaGMyVjBNMTlqYkhWemRHVnlhVzVuWDNKbGMzVnNkSE0ucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzkwODEyODAwfX19XX0_&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEYCIQDO4~F5TAuoYXCQMCwNo-nuuqtzdfeIm~NZVSlAJhAnUAIhAOBriN5PvXmUFzdrgfqFK83~qX5Hh6p3hSSNle3eWrTN)

### Overall comparisons

![Overall classification comparison](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9vdmVyYWxsX2NsYXNzaWZpY2F0aW9uX2NvbXBhcmlzb24.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTl2ZG1WeVlXeHNYMk5zWVhOemFXWnBZMkYwYVc5dVgyTnZiWEJoY21semIyNC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTA4MTI4MDB9fX1dfQ__&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIQC1z-ZfxD~SiXFugd9s41A~TXmF-k62v8u57pV~IU3~CAIgCPpc9n8hdKE3VEoJKab7omfeZbKhyUvG2eLGjMITvkA_)

![Overall regression comparison](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9vdmVyYWxsX3JlZ3Jlc3Npb25fY29tcGFyaXNvbg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTl2ZG1WeVlXeHNYM0psWjNKbGMzTnBiMjVmWTI5dGNHRnlhWE52YmcucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzkwODEyODAwfX19XX0_&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEQCIAFTszPVnDCeuZIVE2yFQvA2CA2e8C-lLJiIqHfbT8L0AiAXEGLZOBiRZlPHopZ65hhdpai1swWHeYIDBcPeChY7Ug__)

![Overall clustering comparison](https://private-us-east-1.manuscdn.com/sessionFile/OVrHM0g0YAuP6yIrdnMg3e/sandbox/uD17z0tm5MU5Bh0ct4ZGI4-images_1788975736916_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9wcm9qZWN0L2Fzc2V0cy9vdmVyYWxsX2NsdXN0ZXJpbmdfY29tcGFyaXNvbg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT1ZySE0wZzBZQXVQNnlJcmRuTWczZS9zYW5kYm94L3VEMTd6MHRtNU1VNUJoMGN0NFpHSTQtaW1hZ2VzXzE3ODg5NzU3MzY5MTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5d2NtOXFaV04wTDJGemMyVjBjeTl2ZG1WeVlXeHNYMk5zZFhOMFpYSnBibWRmWTI5dGNHRnlhWE52YmcucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzkwODEyODAwfX19XX0_&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIQCgVwx9znLkXWhtTwjH-U4k8rxqEuJED89DbFcVYGwu9QIgRoIxBwcTW7-N~zoLAYa600XBifJTfzYd8wgxOJ3Tx18_)

No dimensionality-reduction plot is recorded in the supplied notebook output. The dimensionality-reduction loop records execution status, runtime, and component count, but does not display a chart.

## Installation

The project does not include a `requirements.txt` file. Install the packages imported by the notebook in a Python 3 environment:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn gdown ipython jupyter
```

## Usage

### Google Colab

1. Open `project_for_ml_dinal.ipynb` in Google Colab.
2. Install the dependencies listed above if they are not already available.
3. Replace `MY_DATASET_LINKS` with one or more accessible Google Drive file links, separated by spaces or new lines.
4. Run the notebook cells in order.
5. Review the printed performance report and plots.
6. Find the exported results at `ColabData/custom_datasets_results.csv`.

The notebook uses Colab-style cell annotations such as `#@title` and `#@markdown`, and its recorded run uses a Python 3 kernel.

### Local Jupyter or VS Code

The notebook is standard `.ipynb` Python code and can also be opened in Jupyter or VS Code with the Jupyter extension. The supplied project does not contain a separate VS Code configuration or a recorded VS Code run. Ensure that the Google Drive links are accessible and that the working directory permits creation of the `ColabData/` folder.

## Project Structure

```text
project_for_ml_dinal/
├── project_for_ml_dinal.ipynb
├── README.md
└── assets/
    ├── dataset1_classification_results.png
    ├── dataset1_regression_results.png
    ├── dataset1_clustering_results.png
    ├── dataset2_classification_results.png
    ├── dataset2_regression_results.png
    ├── dataset2_clustering_results.png
    ├── dataset3_classification_results.png
    ├── dataset3_regression_results.png
    ├── dataset3_clustering_results.png
    ├── overall_classification_comparison.png
    ├── overall_regression_comparison.png
    └── overall_clustering_comparison.png
```

`ColabData/` is created at runtime by the notebook and is not part of the supplied upload. The recorded execution also writes `custom_datasets_results.csv` into that directory.

## Limitations and Observations

- The notebook has no `requirements.txt`, license file, author information, or separate source modules.
- Dataset provenance is not specified beyond the Google Drive links embedded in the notebook.
- The default target is `Diabetes_012`; if that column is absent, the code silently uses the last column. In the recorded run, this resulted in `Income` for Datasets 2 and 3.
- The same target is used for classification, regression, and preprocessing. Whether each target is semantically appropriate for all three task types is not documented in the project.
- Models are fitted on at most 1,000 randomly sampled training rows, while evaluation uses the full test split. Therefore, the displayed results are sample-fit benchmark results rather than full-training results.
- The notebook uses one train/test split and does not perform cross-validation or hyperparameter tuning.
- Several algorithms can fail on particular datasets; failures are captured in `results_df`, but the supplied printed report shows successful results only.
- Dimensionality-reduction outputs are recorded as component counts and status records, but are not visualized in the notebook output.
- The imported Calinski–Harabasz metric is not calculated in the execution loop.
- The notebook does not save trained models, preprocessing scalers, or a machine-readable experiment configuration beyond the results CSV.

## Future Improvements

The following are proposed improvements, not features documented as already implemented:

- Add a pinned `requirements.txt` and a license.
- Document each dataset’s original name, schema, feature meanings, and license.
- Validate target semantics separately for classification and regression.
- Replace silent target fallback with an explicit configuration or validation error.
- Add cross-validation, hyperparameter tuning, and confidence intervals.
- Record class distributions and baseline scores.
- Add dimensionality-reduction visualizations and use the imported Calinski–Harabasz metric where appropriate.
- Save fitted models, scalers, and a complete experiment manifest.
- Add automated tests for data loading, preprocessing, and result-schema consistency.

## Technologies

- Python 3
- Jupyter Notebook / Google Colab-compatible notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- gdown
- IPython

## Author

Author information is not specified in the supplied notebook.
