"""
Model Training for Diabetes Prediction

This script implements multiple machine learning models for diabetes prediction,
including training, evaluation, and model persistence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import pickle
import os
import sys

# Import preprocessing functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from preprocess import load_preprocessed_data, preprocess_pipeline

# Import custom plotting functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from custom_functions.plots import scatter_with_regr, plot_residuals, plot_kde
    HAS_CUSTOM_PLOTS = True
except ImportError:
    HAS_CUSTOM_PLOTS = False


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    print("\nTraining Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression training completed!")
    return model


def train_ridge_regression(X_train, y_train, alpha=1.0):
    """
    Train a Ridge Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        alpha: Regularization strength
        
    Returns:
        Trained model
    """
    print(f"\nTraining Ridge Regression (alpha={alpha})...")
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    print("Ridge Regression training completed!")
    return model


def train_lasso_regression(X_train, y_train, alpha=1.0):
    """
    Train a Lasso Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        alpha: Regularization strength
        
    Returns:
        Trained model
    """
    print(f"\nTraining Lasso Regression (alpha={alpha})...")
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    print("Lasso Regression training completed!")
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        random_state: Random seed
        
    Returns:
        Trained model
    """
    print(f"\nTraining Random Forest (n_estimators={n_estimators})...")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print("Random Forest training completed!")
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Gradient Boosting Regressor.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting stages
        random_state: Random seed
        
    Returns:
        Trained model
    """
    print(f"\nTraining Gradient Boosting (n_estimators={n_estimators})...")
    model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print("Gradient Boosting training completed!")
    return model


def train_svr(X_train, y_train, kernel='rbf', C=1.0):
    """
    Train a Support Vector Regressor.
    
    Args:
        X_train: Training features
        y_train: Training target
        kernel: Kernel type
        C: Regularization parameter
        
    Returns:
        Trained model
    """
    print(f"\nTraining SVR (kernel={kernel}, C={C})...")
    model = SVR(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    print("SVR training completed!")
    return model


def train_polynomial_regression(X_train, y_train, degree=3):
    """
    Train a Polynomial Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        degree: Degree of polynomial features
        
    Returns:
        Trained model
    """
    print(f"\nTraining Polynomial Regression (degree={degree})...")
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    print("Polynomial Regression training completed!")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance on training and testing sets.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        model_name: Name of the model for display
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                 scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    metrics = {
        'model_name': model_name,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_rmse': cv_rmse
    }
    
    # Display metrics
    print(f"\n{model_name} Performance:")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Testing RMSE:  {test_rmse:.4f}")
    print(f"  Training R²:   {train_r2:.4f}")
    print(f"  Testing R²:    {test_r2:.4f}")
    print(f"  CV RMSE:       {cv_rmse:.4f}")
    
    return metrics


def plot_predictions(y_true, y_pred, model_name, output_dir='ouputs'):
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model
        output_dir: Directory to save plot
    """
    print(f"Generating plots for {model_name}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Standard actual vs predicted scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_predictions.png"))
    plt.close()
    
    # 2. Integration with custom notebook plots if available
    if HAS_CUSTOM_PLOTS:
        # KDE Plot
        plot_kde(y_true, y_pred)
        plt.title(f'{model_name}: KDE Actual vs Predicted')
        plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_kde.png"))
        plt.close()
        
        # Residuals Plot
        plot_residuals(y_true, y_pred, f'{model_name} Residuals', 'Actual', 'Predicted')
        plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_residuals.png"))
        plt.close()
        
        # Scatter with Regr Line (Custom)
        scatter_with_regr(y_true, y_pred)
        plt.title(f'{model_name}: Actual vs Predicted (Custom)')
        plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_custom_scatter.png"))
        plt.close()


def save_model(model, model_name, output_dir='ouputs'):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model
        output_dir: Directory to save model
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {filepath}")


def load_model(model_name, input_dir='ouputs'):
    """
    Load trained model from disk.
    
    Args:
        model_name: Name of the model
        input_dir: Directory containing model file
        
    Returns:
        Loaded model
    """
    filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    filepath = os.path.join(input_dir, filename)
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from: {filepath}")
    return model


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        
    Returns:
        dict: Dictionary of trained models and their metrics
    """
    print("="*80)
    print("TRAINING MULTIPLE MODELS")
    print("="*80)
    
    models = {}
    results = []
    
    # Linear Regression
    lr = train_linear_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr, X_train, y_train, X_test, y_test, "Linear Regression")
    models['Linear Regression'] = lr
    results.append(lr_metrics)
    plot_predictions(y_test, lr.predict(X_test), "Linear Regression")
    save_model(lr, "Linear Regression")
    
    # Ridge Regression
    ridge = train_ridge_regression(X_train, y_train, alpha=1.0)
    ridge_metrics = evaluate_model(ridge, X_train, y_train, X_test, y_test, "Ridge Regression")
    models['Ridge Regression'] = ridge
    results.append(ridge_metrics)
    plot_predictions(y_test, ridge.predict(X_test), "Ridge Regression")
    save_model(ridge, "Ridge Regression")
    
    # Lasso Regression
    lasso = train_lasso_regression(X_train, y_train, alpha=0.1)
    lasso_metrics = evaluate_model(lasso, X_train, y_train, X_test, y_test, "Lasso Regression")
    models['Lasso Regression'] = lasso
    results.append(lasso_metrics)
    plot_predictions(y_test, lasso.predict(X_test), "Lasso Regression")
    save_model(lasso, "Lasso Regression")
    
    # Random Forest
    rf = train_random_forest(X_train, y_train, n_estimators=100)
    rf_metrics = evaluate_model(rf, X_train, y_train, X_test, y_test, "Random Forest")
    models['Random Forest'] = rf
    results.append(rf_metrics)
    plot_predictions(y_test, rf.predict(X_test), "Random Forest")
    save_model(rf, "Random Forest")
    
    # Gradient Boosting
    gb = train_gradient_boosting(X_train, y_train, n_estimators=100)
    gb_metrics = evaluate_model(gb, X_train, y_train, X_test, y_test, "Gradient Boosting")
    models['Gradient Boosting'] = gb
    results.append(gb_metrics)
    plot_predictions(y_test, gb.predict(X_test), "Gradient Boosting")
    save_model(gb, "Gradient Boosting")
    
    # SVR
    svr = train_svr(X_train, y_train, kernel='rbf', C=1.0)
    svr_metrics = evaluate_model(svr, X_train, y_train, X_test, y_test, "SVR")
    models['SVR'] = svr
    results.append(svr_metrics)
    plot_predictions(y_test, svr.predict(X_test), "SVR")
    save_model(svr, "SVR")
    
    # Polynomial Regression (Added for notebook parity)
    poly = train_polynomial_regression(X_train, y_train, degree=3)
    poly_metrics = evaluate_model(poly, X_train, y_train, X_test, y_test, "Polynomial Regression")
    models['Polynomial Regression'] = poly
    results.append(poly_metrics)
    plot_predictions(y_test, poly.predict(X_test), "Polynomial Regression")
    save_model(poly, "Polynomial Regression")
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(results_df[['model_name', 'test_rmse', 'test_r2', 'cv_rmse']].to_string(index=False))
    
    # Save results
    results_df.to_csv('ouputs/model_comparison.csv', index=False)
    print("\nModel comparison saved to 'ouputs/model_comparison.csv'")
    
    # Find best model
    best_model_name = results_df.loc[results_df['test_r2'].idxmax(), 'model_name']
    print(f"\nBest model based on R²: {best_model_name}")
    
    return models, results_df


def main():
    """
    Main function to run model training pipeline.
    """
    print("="*80)
    print("DIABETES PREDICTION - MODEL TRAINING")
    print("="*80)
    
    # Load preprocessed data
    try:
        X_train, X_test, y_train, y_test, scaler = load_preprocessed_data()
    except FileNotFoundError:
        print("Preprocessed data not found. Running preprocessing pipeline...")
        X_train, X_test, y_train, y_test, scaler, _ = preprocess_pipeline()
    
    # Train all models
    models, results = train_all_models(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return models, results


if __name__ == "__main__":
    models, results = main()
