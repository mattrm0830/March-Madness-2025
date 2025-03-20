import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import optuna
import matplotlib.pyplot as plt




def advanced_time_series_cv(X, y, seasons_array, splits, n_optuna_trials=10):
    """
    Enhanced time-series cross-validation with hyperparameter tuning using updated Optuna methods
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - seasons_array: Array of seasons corresponding to each row
    - splits: Time series splits
    - n_optuna_trials: Number of Optuna optimization trials
    """
    results = []
    
    # Optuna objective function for hyperparameter optimization
    def objective(trial):
        # Suggest hyperparameters using updated Optuna methods
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'n_estimators': 1000,  # Keep consistent with original
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'early_stopping_rounds': 50
        }
        
        # Best model metrics to track
        best_metrics = {
            'val_log_loss': float('inf'),
            'val_accuracy': 0,
            'val_auc': 0,
            'best_iteration': 0
        }
        
        # Perform time-series cross-validation
        for train_seasons, val_season in splits:
            # Create train/validation masks
            train_mask = np.isin(seasons_array, train_seasons)
            val_mask = seasons_array == val_season
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            
            # Create and train the model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predictions and metrics
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            val_log_loss = log_loss(y_val, y_pred_proba)
            val_accuracy = accuracy_score(y_val, y_pred)
            val_auc = roc_auc_score(y_val, y_pred_proba)
            
            # Update best metrics
            if val_log_loss < best_metrics['val_log_loss']:
                best_metrics.update({
                    'val_log_loss': val_log_loss,
                    'val_accuracy': val_accuracy,
                    'val_auc': val_auc,
                    'best_iteration': model.best_iteration
                })
        
        # Return the primary metric for Optuna to optimize (log loss)
        return best_metrics['val_log_loss']
    
    # Perform Optuna hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_optuna_trials)
    
    # Print best hyperparameters
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    # Final training with best hyperparameters
    best_params = study.best_params
    best_params.update({
        'n_estimators': 5000,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    })
    
    # Detailed results for each split
    for i, (train_seasons, val_season) in enumerate(splits):
        # Create train/validation masks
        train_mask = np.isin(seasons_array, train_seasons)
        val_mask = seasons_array == val_season
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        # Train the model with best parameters
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        loss = log_loss(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # Store results
        results.append({
            'split': i+1,
            'train_seasons': train_seasons,
            'val_season': val_season,
            'model': model,
            'log_loss': loss,
            'accuracy': acc,
            'auc': auc,
            'best_iteration': model.best_iteration,
            'feature_importance': model.feature_importances_
        })
        
        print(f"Validation results - Log Loss: {loss:.4f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    return results

# Example usage (similar to original script)
def main():
    # Load data
    matchups_df = pd.read_csv('data_processing/matchup_features.csv')
    
    # Get unique seasons in chronological order
    seasons = np.sort(matchups_df['Season'].unique())
    print(f"Available seasons: {seasons}")
    
    # Create time-series splits
    splits = []
    min_train_seasons = 3  # Start with at least 3 seasons of training data
    
    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]
        splits.append((train_seasons, val_season))
    
    # Select features for the model
    features = [col for col in matchups_df.columns if col.endswith('_diff') or col == 'IsTournament']
    X = matchups_df[features].fillna(0).values
    y = matchups_df['Result'].values
    seasons_array = matchups_df['Season'].values
    
    # Run advanced time-series cross-validation
    cv_results = advanced_time_series_cv(X, y, seasons_array, splits)
    
    # Summarize results
    print("\nSummary of cross-validation results:")
    for result in cv_results:
        print(f"Split {result['split']}: Train on {result['train_seasons']}, validate on {result['val_season']}")
        print(f"  Log Loss: {result['log_loss']:.4f}, Accuracy: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}")
    
    # Calculate average metrics
    avg_loss = np.mean([result['log_loss'] for result in cv_results])
    avg_acc = np.mean([result['accuracy'] for result in cv_results])
    avg_auc = np.mean([result['auc'] for result in cv_results])
    print(f"\nAverage metrics - Log Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, AUC: {avg_auc:.4f}")
    
    # Find the best model (lowest log loss)
    best_idx = np.argmin([result['log_loss'] for result in cv_results])
    best_model = cv_results[best_idx]['model']
    print(f"Best model from split {cv_results[best_idx]['split']} (validated on {cv_results[best_idx]['val_season']})")
    print(f"Best model metrics - Log Loss: {cv_results[best_idx]['log_loss']:.4f}, Accuracy: {cv_results[best_idx]['accuracy']:.4f}, AUC: {cv_results[best_idx]['auc']:.4f}")
    
    # Feature importance visualization
  
    
    plt.figure(figsize=(10, 6))
    feature_importance = cv_results[best_idx]['feature_importance']
    feature_names = features
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, [feature_names[i] for i in sorted_idx])
    plt.title('Feature Importance in Best XGBoost Model')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()