
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, average_precision_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(filepath='creditcard.csv'):
    """
    Loads the dataset from CSV or generates a synthetic one if the file doesn't exist.
    """
    if os.path.exists(filepath):
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
    else:
        print(f"File {filepath} not found. Generating synthetic dataset for demonstration...")
        # Generate synthetic data mimicking credit card fraud data. 
        # Using 30 features to mimic V1-V28, Time, Amount.
        X, y = make_classification(n_samples=10000, n_features=30, n_informative=24, n_redundant=2,
                                   n_clusters_per_class=1, weights=[0.99], random_state=42)
        df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'])
        df['Class'] = y
        # Adjust Time and Amount to look more realistic
        df['Time'] = df['Time'].abs() * 1000
        df['Amount'] = df['Amount'].abs() * 100
        
    print(f"Dataset Shape: {df.shape}")
    return df

def perform_eda(df):
    """
    Performs basic Exploratory Data Analysis.
    """
    print("\n--- Exploratory Data Analysis ---")
    print("Class Distribution:")
    print(df['Class'].value_counts())
    
    # Visualizing class imbalance
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df, palette='viridis')
    plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print("Saved class distribution plot to 'class_distribution.png'")

def preprocess_data(df):
    """
    Scales Amount and Time, and splits the data.
    """
    print("\n--- Preprocessing ---")
    
    # 1. Scale 'Amount' and 'Time'
    # RobustScaler is less prone to outliers
    rob_scaler = RobustScaler()
    
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Separating Feature and Target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 2. Stratified Split
    print("Splitting data (80/20 split with stratification)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """
    Applies SMOTE to the training data only.
    """
    print("\n--- Handling Imbalance (SMOTE) ---")
    print(f"Original train counts: {y_train.value_counts().to_dict()}")
    
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"Resampled train counts: {y_train_res.value_counts().to_dict()}")
    return X_train_res, y_train_res

def train_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates Logistic Regression, Decision Tree, and Random Forest.
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    print("\n--- Model Training & Evaluation ---")
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        # Get probabilities for AUPRC if available
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = clf.decision_function(X_test)
        
        # Metrics
        print(f"--- {name} Results ---")
        print(classification_report(y_test, y_pred))
        
        # AUPRC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auprc_score = auc(recall, precision)
        print(f"AUPRC: {auprc_score:.4f}")
        
        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} CM (AUPRC: {auprc_score:.2f})')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        filename = f'cm_{name.replace(" ", "_").lower()}.png'
        plt.savefig(filename)
        print(f"Saved confusion matrix to '{filename}'")

def analysis_summary():
    """
    Prints the required analysis and summary.
    """
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    print("\n1. Metrics Importance (Precision vs Recall):")
    print("""
    For Fraud Detection, RECALL is generally considered more critical than Precision, but there is a trade-off.
    - HIGH RECALL: Means we catch most fraud cases (minimizing False Negatives). A False Negative (missing a fraud) is very costly.
    - HIGH PRECISION: Means we don't annoy legitimate customers (minimizing False Positives). A False Positive (flagging legit txn) causes friction.
    
    Ideally, we aim for the highest F1-Score or AUPRC. However, if we must choose:
    - Choose RECALL if the cost of fraud is extremely high (e.g., millions lost per case).
    - Choose PRECISION if the cost of verifying a transaction is high or customer churn from blocked cards is a major concern.
    
    Usually, we want to maximize Recall while keeping Precision at an acceptable level.
    """)
    
    print("\n2. Random Forest vs Logistic Regression:")
    print("""
    - RANDOM FOREST often performs better on this task because:
        a) It handles non-linear relationships between features better than Logistic Regression.
        b) It is an ensemble method (bagging), which reduces variance and makes it robust to outliers and noise.
        c) It handles imbalanced data generally better than single trees or simple linear models, especially with class weighting (though we used SMOTE here).
    
    - LOGISTIC REGRESSION might perform worse if the decision boundary is highly non-linear, but it is faster and more interpretable.
    """)

if __name__ == "__main__":
    # 1. Load Data
    df = load_data('creditcard.csv')
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 4. Handle Imbalance (SMOTE)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # 5. Model Development & Evaluation
    train_evaluate_models(X_train_res, y_train_res, X_test, y_test)
    
    # 6. Analysis
    analysis_summary()
