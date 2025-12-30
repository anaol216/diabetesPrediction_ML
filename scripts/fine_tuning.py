"""
Hyperparameter Fine-Tuning for Diabetes Prediction Models

This script performs hyperparameter optimization using RandomizedSearchCV
and GridSearchCV to find the best model configurations.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import loguniform, uniform, randint
import pickle
import os
import sys

# Import preprocessing functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_preprocessed_data, preprocess_pipeline


def tune_random_forest(X_train, y_train, n_iter=50, cv=5, random_state=42):
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Best estimator and search results
    """
    print("\n" + "="*80)
    print("TUNING RANDOM FOREST HYPERPARAMETERS")
    print("="*80)
    
    # Define parameter distributions
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    # Create base model
    rf = RandomForestRegressor(random_state=random_state)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='neg_mean_squared_error', n_jobs=-1,
        random_state=random_state, verbose=2
    )
    
    print(f"\nSearching through {n_iter} parameter combinations...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.4f}")
    
    return random_search.best_estimator_, random_search


def tune_gradient_boosting(X_train, y_train, n_iter=50, cv=5, random_state=42):
    """
    Tune Gradient Boosting hyperparameters using RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Best estimator and search results
    """
    print("\n" + "="*80)
    print("TUNING GRADIENT BOOSTING HYPERPARAMETERS")
    print("="*80)
    
    # Define parameter distributions
    param_dist = {
        'n_estimators': randint(50, 300),
        'learning_rate': loguniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    gb = GradientBoostingRegressor(random_state=random_state)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        gb, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='neg_mean_squared_error', n_jobs=-1,
        random_state=random_state, verbose=2
    )
    
    print(f"\nSearching through {n_iter} parameter combinations...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.4f}")
    
    return random_search.best_estimator_, random_search


def tune_svr(X_train, y_train, n_iter=50, cv=5, random_state=42):
    """
    Tune SVR hyperparameters using RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Best estimator and search results
    """
    print("\n" + "="*80)
    print("TUNING SVR HYPERPARAMETERS")
    print("="*80)
    
    # Define parameter distributions
    param_dist = {
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'C': loguniform(0.1, 100),
        'gamma': ['scale', 'auto'] + list(loguniform(0.001, 1).rvs(5, random_state=random_state)),
        'epsilon': loguniform(0.01, 1)
    }
    
    # Create base model
    svr = SVR()
    
    # Randomized search
    random_search = RandomizedSearchCV(
        svr, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='neg_mean_squared_error', n_jobs=-1,
        random_state=random_state, verbose=2
    )
    
    print(f"\nSearching through {n_iter} parameter combinations...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.4f}")
    
    return random_search.best_estimator_, random_search


def tune_ridge(X_train, y_train, cv=5):
    """
    Tune Ridge Regression hyperparameters using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        Best estimator and search results
    """
    print("\n" + "="*80)
    print("TUNING RIDGE REGRESSION HYPERPARAMETERS")
    print("="*80)
    
    # Define parameter grid
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    }
    
    # Create base model
    ridge = Ridge()
    
    # Grid search
    grid_search = GridSearchCV(
        ridge, param_grid=param_grid, cv=cv,
        scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
    )
    
    print(f"\nSearching through {len(param_grid['alpha'])} alpha values...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_, grid_search


def tune_lasso(X_train, y_train, cv=5):
    """
    Tune Lasso Regression hyperparameters using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        Best estimator and search results
    """
    print("\n" + "="*80)
    print("TUNING LASSO REGRESSION HYPERPARAMETERS")
    print("="*80)
    
    # Define parameter grid
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    # Create base model
    lasso = Lasso()
    
    # Grid search
    grid_search = GridSearchCV(
        lasso, param_grid=param_grid, cv=cv,
        scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
    )
    
    print(f"\nSearching through {len(param_grid['alpha'])} alpha values...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_, grid_search


def evaluate_tuned_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Evaluate tuned model performance.
    
    Args:
        model: Tuned model
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        model_name: Name of the model
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating tuned {model_name}...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Testing RMSE:  {test_rmse:.4f}")
    print(f"  Training R²:   {train_r2:.4f}")
    print(f"  Testing R²:    {test_r2:.4f}")
    
    return metrics


def save_tuned_model(model, model_name, output_dir='ouputs'):
    """
    Save tuned model to disk.
    
    Args:
        model: Tuned model
        model_name: Name of the model
        output_dir: Directory to save model
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"tuned_{model_name.lower().replace(' ', '_')}_model.pkl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Tuned model saved to: {filepath}")


def tune_all_models(X_train, y_train, X_test, y_test, n_iter=50, cv=5, random_state=42):
    """
    Tune all models and compare performance.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        n_iter: Number of iterations for randomized search
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        dict: Dictionary of tuned models and results
    """
    print("="*80)
    print("HYPERPARAMETER TUNING FOR ALL MODELS")
    print("="*80)
    
    tuned_models = {}
    results = []
    
    # Tune Ridge
    ridge_tuned, ridge_search = tune_ridge(X_train, y_train, cv=cv)
    ridge_metrics = evaluate_tuned_model(ridge_tuned, X_train, y_train, X_test, y_test, "Ridge")
    tuned_models['Ridge'] = ridge_tuned
    results.append(ridge_metrics)
    save_tuned_model(ridge_tuned, "Ridge")
    
    # Tune Lasso
    lasso_tuned, lasso_search = tune_lasso(X_train, y_train, cv=cv)
    lasso_metrics = evaluate_tuned_model(lasso_tuned, X_train, y_train, X_test, y_test, "Lasso")
    tuned_models['Lasso'] = lasso_tuned
    results.append(lasso_metrics)
    save_tuned_model(lasso_tuned, "Lasso")
    
    # Tune Random Forest
    rf_tuned, rf_search = tune_random_forest(X_train, y_train, n_iter=n_iter, cv=cv, random_state=random_state)
    rf_metrics = evaluate_tuned_model(rf_tuned, X_train, y_train, X_test, y_test, "Random Forest")
    tuned_models['Random Forest'] = rf_tuned
    results.append(rf_metrics)
    save_tuned_model(rf_tuned, "Random Forest")
    
    # Tune Gradient Boosting
    gb_tuned, gb_search = tune_gradient_boosting(X_train, y_train, n_iter=n_iter, cv=cv, random_state=random_state)
    gb_metrics = evaluate_tuned_model(gb_tuned, X_train, y_train, X_test, y_test, "Gradient Boosting")
    tuned_models['Gradient Boosting'] = gb_tuned
    results.append(gb_metrics)
    save_tuned_model(gb_tuned, "Gradient Boosting")
    
    # Tune SVR
    svr_tuned, svr_search = tune_svr(X_train, y_train, n_iter=n_iter, cv=cv, random_state=random_state)
    svr_metrics = evaluate_tuned_model(svr_tuned, X_train, y_train, X_test, y_test, "SVR")
    tuned_models['SVR'] = svr_tuned
    results.append(svr_metrics)
    save_tuned_model(svr_tuned, "SVR")
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("TUNED MODELS COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('ouputs/tuned_model_comparison.csv', index=False)
    print("\nTuned model comparison saved to 'ouputs/tuned_model_comparison.csv'")
    
    # Find best model
    best_model_name = results_df.loc[results_df['test_r2'].idxmax(), 'model_name']
    best_model = tuned_models[best_model_name]
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*80}")
    
    # Save best model separately
    save_tuned_model(best_model, "best_model")
    
    return tuned_models, results_df, best_model


def main():
    """
    Main function to run hyperparameter tuning pipeline.
    """
    print("="*80)
    print("DIABETES PREDICTION - HYPERPARAMETER TUNING")
    print("="*80)
    
    # Load preprocessed data
    try:
        X_train, X_test, y_train, y_test, scaler = load_preprocessed_data()
    except FileNotFoundError:
        print("Preprocessed data not found. Running preprocessing pipeline...")
        X_train, X_test, y_train, y_test, scaler, _ = preprocess_pipeline()
    
    # Tune all models
    tuned_models, results, best_model = tune_all_models(
        X_train, y_train, X_test, y_test,
        n_iter=30,  # Reduced for faster execution
        cv=5,
        random_state=42
    )
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return tuned_models, results, best_model


if __name__ == "__main__":
    tuned_models, results, best_model = main()
