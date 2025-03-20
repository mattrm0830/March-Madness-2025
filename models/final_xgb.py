"""
Final Models for xgboost.
Best Hyperparameters:
    learning_rate: 0.13621699419719052
    max_depth: 4
    min_child_weight: 9
    subsample: 0.8497044005595025
    colsample_bytree: 0.8400075675185479
    gamma: 3.5464291595313e-07
"""


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt




def load_data(train_path, test_path):
    """
    Load and prepare training and test data for modeling
    
    Parameters:
    - train_path: Path to training data CSV
    - test_path: Path to test data CSV
    
    Returns:
    - X_train: Training features
    - y_train: Training target variable
    - X_test: Test features
    - test_ids: Test IDs for submission
    - feature_names: List of feature names
    """
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)  # Fixed: was using train_path instead of test_path
    
    print(f"Loaded {train.shape[0]} training examples and {test.shape[0]} test examples")
    
    # Identify non-feature columns
    non_feature_cols = ['ID', 'Season', 'Team1ID', 'Team2ID', 'Gender', 'Outcome']
    
    # Identify the target column
    target_col = 'Outcome'  # Using 'Outcome' instead of 'Result' based on our data structure
    
    # Extract features and target
    feature_cols = [col for col in train.columns if col not in non_feature_cols]
    
    # Check if target column exists
    if target_col not in train.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    
    # Handle NaN values - can use different strategies depending on your data
    train[feature_cols] = train[feature_cols].fillna(0)
    test[feature_cols] = test[feature_cols].fillna(0)
    
    # Extract X and y for training
    X_train = train[feature_cols]
    y_train = train[target_col]
    
    # Extract X for testing and IDs for submission
    X_test = test[feature_cols]
    test_ids = test['ID'] if 'ID' in test.columns else None
    
    print(f"Using {len(feature_cols)} features for modeling")
    
    return X_train, y_train, X_test, test_ids, feature_cols

def train_final_model(X_train, y_train):
    """
    Train final XGBoost model with best parameters
    
    Parameters:
    - X_train: Training features
    - y_train: Training target
    
    Returns:
    - Trained XGBoost model
    """
    # Best parameters from Optuna (REPLACE WITH YOUR ACTUAL PARAMETERS)
    best_params = {
        'learning_rate': 0.13621699419719052,
        'max_depth': 4,
        'min_child_weight': 9,
        'subsample': 0.8497044005595025,
        'colsample_bytree': 0.8400075675185479,
        'gamma': 3.5464291595313e-07,
        'n_estimators': 1000,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    # Train final model
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_train, y_train):
    """
    Evaluate the trained model
    
    Parameters:
    - model: Trained XGBoost model
    - X_train: Training features
    - y_train: Training target
    
    Returns:
    - Dictionary of performance metrics
    """
    # Predictions
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    y_pred = model.predict(X_train)
    
    # Calculate metrics
    metrics = {
        'log_loss': log_loss(y_train, y_pred_proba),
        'accuracy': accuracy_score(y_train, y_pred),
        'auc': roc_auc_score(y_train, y_pred_proba)
    }
    
    return metrics

def generate_predictions(model, X_test):
    """
    Generate probability predictions for test data
    
    Parameters:
    - model: Trained XGBoost model
    - X_test: Test features
    
    Returns:
    - Probability predictions
    """
    return model.predict_proba(X_test)[:, 1]

def main():
    # Paths to your data files (adjust as needed)
    train_path = 'data/model/training_data.csv'
    test_path = 'data/model/test_data.csv'  
    
    # Load and prepare data
    X_train, y_train, X_test, test_ids, features = load_data(train_path, test_path)
    
    # Train final model
    final_model = train_final_model(X_train, y_train)
    
    # Evaluate model
    model_metrics = evaluate_model(final_model, X_train, y_train)
    print("Model Performance Metrics:")
    for metric, value in model_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate predictions
    predictions = generate_predictions(final_model, X_test)

    # Save predictions
    predictions_df = pd.DataFrame({
        'ID': test_ids,  # Use the test_ids already returned from load_data
        'Pred': predictions  # Note: 'Pred' is often the expected column name for Kaggle
    })
    predictions_df.to_csv('data/model/march_madness_predictions.csv', index=False)
        
    print(f"Saved predictions to ../data/march_madness_predictions.csv")
    
 
    
    plt.figure(figsize=(10, 6))
    feature_importance = final_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, [features[i] for i in sorted_idx])
    plt.title('Feature Importance in Final XGBoost Model')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()