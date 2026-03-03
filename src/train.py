from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model(X_train, y_train, algorithm='RandomForest'):
    # Select the algorithm based on the parameter
    if algorithm == 'LogisticRegression':
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm == 'DecisionTree':
        clf = DecisionTreeClassifier(random_state=42)
    else:
        # Default to Random Forest (usually the best performer)
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_model(clf, filepath='models/fraud_model.joblib'):
    joblib.dump(clf, filepath)