from src.data import load_data
from src.preprocessing import preprocess_data, apply_smote
from src.train import train_model, evaluate_model, save_model
from src.utils import logger
import sys

def main():
    logger.info("Starting Task 2: Credit Card Fraud Detection Pipeline")
    
    try:
        # 1. Load Data
        df = load_data()
        
        # 2. Preprocess (Handles the new text columns)
        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # 3. Handle Imbalance with SMOTE
        logger.info("Applying SMOTE...")
        X_train_res, y_train_res = apply_smote(X_train, y_train)
        
        # 4. Experiment with Algorithms
        algorithms = ['LogisticRegression', 'DecisionTree', 'RandomForest']
        best_model = None
        
        for algo in algorithms:
            logger.info(f"\n--- Training {algo} ---")
            clf = train_model(X_train_res, y_train_res, algorithm=algo)
            evaluate_model(clf, X_test, y_test)
            
            # Save Random Forest as the final API model
            if algo == 'RandomForest':
                best_model = clf
                
        # 5. Save Model
        save_model(best_model)
        logger.info("Final model saved successfully! Ready for the API.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()