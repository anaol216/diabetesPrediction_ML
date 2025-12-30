"""
Exploratory Data Analysis (EDA) for Diabetes Dataset

This script performs comprehensive exploratory data analysis on the diabetes dataset,
including data loading, statistical analysis, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
import os
import sys

# Add custom_functions to path if it exists
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from custom_functions.plots import *
except ImportError:
    print("Custom plotting functions not found. Using standard matplotlib/seaborn.")


def load_data():
    """
    Load the diabetes dataset from sklearn.
    
    Returns:
        tuple: (X, y, feature_names, target_name, description)
    """
    diab = load_diabetes()
    X = diab['data']
    y = diab['target']
    feature_names = diab['feature_names']
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"\nFeatures: {feature_names}")
    
    return X, y, feature_names, diab.get('DESCR', ''), diab


def create_dataframe(X, y, feature_names):
    """
    Create a pandas DataFrame from the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        
    Returns:
        pd.DataFrame: Combined dataframe with features and target
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def display_basic_info(df):
    """
    Display basic information about the dataset.
    
    Args:
        df: pandas DataFrame
    """
    print("\n" + "="*80)
    print("DATASET BASIC INFORMATION")
    print("="*80)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nData Types:")
    print(df.dtypes)


def analyze_target_distribution(df):
    """
    Analyze and visualize the target variable distribution.
    
    Args:
        df: pandas DataFrame
    """
    print("\n" + "="*80)
    print("TARGET VARIABLE ANALYSIS")
    print("="*80)
    
    print(f"\nTarget Statistics:")
    print(f"Mean: {df['target'].mean():.2f}")
    print(f"Median: {df['target'].median():.2f}")
    print(f"Std Dev: {df['target'].std():.2f}")
    print(f"Min: {df['target'].min():.2f}")
    print(f"Max: {df['target'].max():.2f}")
    
    # Visualize target distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['target'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Target Variable')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['target'])
    plt.ylabel('Target Value')
    plt.title('Boxplot of Target Variable')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ouputs/target_distribution.png', dpi=300, bbox_inches='tight')
    print("\nTarget distribution plot saved to 'ouputs/target_distribution.png'")
    plt.show()


def analyze_features(df):
    """
    Analyze feature distributions and relationships.
    
    Args:
        df: pandas DataFrame
    """
    print("\n" + "="*80)
    print("FEATURE ANALYSIS")
    print("="*80)
    
    feature_cols = [col for col in df.columns if col != 'target']
    
    # Feature distributions
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col], bins=20, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ouputs/feature_distributions.png', dpi=300, bbox_inches='tight')
    print("\nFeature distributions plot saved to 'ouputs/feature_distributions.png'")
    plt.show()


def correlation_analysis(df):
    """
    Perform correlation analysis and create heatmap.
    
    Args:
        df: pandas DataFrame
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Display correlations with target
    target_corr = corr_matrix['target'].sort_values(ascending=False)
    print("\nCorrelations with Target Variable:")
    print(target_corr)
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ouputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nCorrelation heatmap saved to 'ouputs/correlation_heatmap.png'")
    plt.show()


def feature_vs_target_analysis(df):
    """
    Analyze relationship between each feature and target.
    
    Args:
        df: pandas DataFrame
    """
    print("\n" + "="*80)
    print("FEATURE vs TARGET ANALYSIS")
    print("="*80)
    
    feature_cols = [col for col in df.columns if col != 'target']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_cols):
        axes[idx].scatter(df[col], df['target'], alpha=0.5)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Target')
        axes[idx].set_title(f'{col} vs Target')
        axes[idx].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df[col], df['target'], 1)
        p = np.poly1d(z)
        axes[idx].plot(df[col], p(df[col]), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('ouputs/features_vs_target.png', dpi=300, bbox_inches='tight')
    print("\nFeatures vs target plot saved to 'ouputs/features_vs_target.png'")
    plt.show()


def main():
    """
    Main function to run all EDA steps.
    """
    print("="*80)
    print("DIABETES DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Load data
    X, y, feature_names, description, diab_dict = load_data()
    
    # Print dataset description
    if description:
        print("\n" + "="*80)
        print("DATASET DESCRIPTION")
        print("="*80)
        print(description)
    
    # Create dataframe
    df = create_dataframe(X, y, feature_names)
    
    # Perform analyses
    display_basic_info(df)
    analyze_target_distribution(df)
    analyze_features(df)
    correlation_analysis(df)
    feature_vs_target_analysis(df)
    
    print("\n" + "="*80)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nAll visualizations have been saved to the 'ouputs' folder.")
    
    return df


if __name__ == "__main__":
    df = main()
