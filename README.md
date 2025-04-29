# Credit Card Fraud Detection

This project focuses on detecting fraudulent transactions in a credit card dataset using both traditional machine learning algorithms and a deep learning model.

## Dataset

The dataset used is `creditcard.csv`, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.

### Features:
- **V1 to V28**: Result of a PCA transformation to protect sensitive information.
- **Time**: Seconds elapsed between this transaction and the first transaction in the dataset.
- **Amount**: Transaction amount.
- **Class**: 1 for fraud, 0 for valid transaction.

## Exploratory Data Analysis

- Descriptive statistics of fraudulent and valid transactions were generated.
- Correlation heatmaps were plotted to understand feature relationships.
- An outlier fraction metric was calculated to understand class imbalance.

## Data Preprocessing

- Missing values in features and target variables were handled using `SimpleImputer`.
- Feature scaling was applied using `StandardScaler`.
- Data was split into training and testing sets using `train_test_split` with stratification.

## Machine Learning Models

### 1. Logistic Regression
A baseline model used for binary classification.

### 2. Decision Tree
A non-parametric supervised learning method.

### 3. Support Vector Machine (SVM)
A robust model especially useful for high-dimensional datasets.

### 4. Random Forest Classifier
An ensemble method combining multiple decision trees.

## Deep Learning Model

A sequential neural network model using TensorFlow/Keras with the following structure:
- Dense layer with 30 units (ReLU)
- Dense layer with 16 units (ReLU)
- Dense layer with 8 units (ReLU)
- Output layer with 1 unit (Sigmoid)

### Compilation & Training
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy
- Epochs: 10
- Batch Size: 32
- Validation Split: 20%

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient
- Confusion Matrix

Each model was evaluated on the above metrics to determine its performance in fraud detection.

## Visualization

- Correlation heatmap of features
- Confusion matrix for the deep learning model

## Requirements

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- tensorflow

## How to Run

1. Clone the repository or download the files.
2. Ensure all dependencies are installed (use `pip install -r requirements.txt`).
3. Run the notebook or Python script to process the data, train the models, and evaluate their performance.

## Author

This project was developed as part of a machine learning and deep learning exercise for binary classification problems using real-world financial data.

