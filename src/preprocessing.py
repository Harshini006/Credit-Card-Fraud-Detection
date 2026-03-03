import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    # 1. Drop columns that don't help predict fraud (names, IDs, locations)
    cols_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 
                    'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 
                    'long', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long']
    
    # Only drop columns if they actually exist in the dataframe
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # 2. Encode useful text columns into numbers
    le = LabelEncoder()
    if 'gender' in df.columns:
        df['gender'] = le.fit_transform(df['gender']) # Converts 'M'/'F' to 0/1
    if 'category' in df.columns:
        df['category'] = le.fit_transform(df['category']) # Converts categories to numbers

    # 3. Separate features (X) and Target (y)
    # The new dataset uses 'is_fraud' instead of 'Class'
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # 4. Scale the 'amt' (Amount) column so large purchases don't break the model
    scaler = StandardScaler()
    if 'amt' in X.columns:
        X['amt'] = scaler.fit_transform(X[['amt']])

    # 5. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    # SMOTE handles the class imbalance by creating synthetic fraud cases
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res