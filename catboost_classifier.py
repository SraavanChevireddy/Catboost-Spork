import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import time


class CatBoostAnalyzer:
    """
    A comprehensive CatBoost model implementation demonstrating its key advantages.
    """
    
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.training_time = None
        
    def load_sample_data(self, task_type='classification'):
        """
        Load sample datasets for demonstration.
        
        Args:
            task_type (str): 'classification' or 'regression'
            
        Returns:
            tuple: X, y datasets
        """
        if task_type == 'classification':
            # Load breast cancer dataset
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = data.target
            
            # Add some categorical features to demonstrate CatBoost's strength
            X['feature_group'] = pd.cut(X['mean radius'], bins=3, labels=['small', 'medium', 'large'])
            X['texture_category'] = pd.cut(X['mean texture'], bins=4, labels=['smooth', 'moderate', 'rough', 'very_rough'])
            
        else:  # regression
            # Create a synthetic dataset with categorical features
            np.random.seed(42)
            n_samples = 1000
            
            # Numerical features
            X_num = np.random.randn(n_samples, 5)
            
            # Categorical features
            categories = ['A', 'B', 'C', 'D']
            cat1 = np.random.choice(categories, n_samples)
            cat2 = np.random.choice(['Type1', 'Type2', 'Type3'], n_samples)
            
            # Create target with categorical influence
            y = (X_num[:, 0] * 2 + X_num[:, 1] * 1.5 + 
                 (cat1 == 'A').astype(int) * 3 + 
                 (cat2 == 'Type1').astype(int) * 2 + 
                 np.random.normal(0, 0.5, n_samples))
            
            X = pd.DataFrame(X_num, columns=[f'num_feature_{i}' for i in range(5)])
            X['category_1'] = cat1
            X['category_2'] = cat2
            
        return X, y
    
    def train_classifier(self, X, y, categorical_features=None):
        """
        Train CatBoost classifier demonstrating key advantages.
        
        Args:
            X: Feature matrix
            y: Target vector
            categorical_features: List of categorical feature names
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize CatBoost classifier with optimal parameters
        self.classifier = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            cat_features=categorical_features,
            verbose=False,
            random_seed=42,
            early_stopping_rounds=50
        )
        
        # Time the training process
        start_time = time.time()
        
        # Train the model - CatBoost handles categorical features automatically
        self.classifier.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False
        )
        
        self.training_time = time.time() - start_time
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"CatBoost Classifier Results:")
        print(f"Training Time: {self.training_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Best Iteration: {self.classifier.get_best_iteration()}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def train_regressor(self, X, y, categorical_features=None):
        """
        Train CatBoost regressor demonstrating key advantages.
        
        Args:
            X: Feature matrix
            y: Target vector
            categorical_features: List of categorical feature names
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize CatBoost regressor
        self.regressor = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            cat_features=categorical_features,
            verbose=False,
            random_seed=42,
            early_stopping_rounds=50
        )
        
        # Time the training process
        start_time = time.time()
        
        # Train the model
        self.regressor.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False
        )
        
        self.training_time = time.time() - start_time
        
        # Make predictions
        y_pred = self.regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"CatBoost Regressor Results:")
        print(f"Training Time: {self.training_time:.2f} seconds")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Best Iteration: {self.regressor.get_best_iteration()}")
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def plot_feature_importance(self, model_type='classifier'):
        """
        Plot feature importance to demonstrate CatBoost's interpretability.
        
        Args:
            model_type (str): 'classifier' or 'regressor'
        """
        model = self.classifier if model_type == 'classifier' else self.regressor
        
        if model is None:
            print(f"No {model_type} model trained yet!")
            return
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        feature_names = model.feature_names_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'CatBoost {model_type.title()} Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def demonstrate_categorical_handling(self):
        """
        Demonstrate CatBoost's superior categorical feature handling.
        """
        print("=" * 60)
        print("DEMONSTRATING CATBOOST'S CATEGORICAL FEATURE HANDLING")
        print("=" * 60)
        
        # Load classification data with categorical features
        X, y = self.load_sample_data('classification')
        
        # Identify categorical features
        categorical_features = ['feature_group', 'texture_category']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Categorical features: {categorical_features}")
        print(f"Categorical feature unique values:")
        for cat_feat in categorical_features:
            print(f"  {cat_feat}: {X[cat_feat].unique()}")
        
        # Train model with categorical features
        results = self.train_classifier(X, y, categorical_features)
        
        print("\nKey Advantage: CatBoost handles categorical features natively")
        print("- No need for manual encoding (Label/One-Hot)")
        print("- Automatic optimal encoding during training")
        print("- Reduces preprocessing complexity")
        print("- Better performance on high-cardinality categories")
        
        return results


def main():
    """
    Main function demonstrating CatBoost capabilities and advantages.
    """
    analyzer = CatBoostAnalyzer()
    
    print("CatBoost Model Implementation and Advantages Demonstration")
    print("=" * 60)
    
    # Demonstrate categorical feature handling
    analyzer.demonstrate_categorical_handling()
    
    print("\n" + "=" * 60)
    print("REGRESSION EXAMPLE WITH CATEGORICAL FEATURES")
    print("=" * 60)
    
    # Regression example
    X_reg, y_reg = analyzer.load_sample_data('regression')
    categorical_features_reg = ['category_1', 'category_2']
    
    print(f"Regression dataset shape: {X_reg.shape}")
    print(f"Categorical features: {categorical_features_reg}")
    
    analyzer.train_regressor(X_reg, y_reg, categorical_features_reg)
    
    print("\n" + "=" * 60)
    print("KEY CATBOOST ADVANTAGES SUMMARY")
    print("=" * 60)
    
    advantages = [
        "1. NATIVE CATEGORICAL FEATURE SUPPORT:",
        "   - No preprocessing required for categorical variables",
        "   - Automatic optimal encoding during training",
        "   - Superior performance on high-cardinality categories",
        "",
        "2. OVERFITTING RESISTANCE:",
        "   - Built-in regularization techniques",
        "   - Ordered boosting prevents target leakage",
        "   - Early stopping prevents overfitting",
        "",
        "3. FAST TRAINING AND PREDICTION:",
        "   - GPU acceleration support",
        "   - Efficient gradient computation",
        "   - Optimized for both speed and accuracy",
        "",
        "4. ROBUST DEFAULT PARAMETERS:",
        "   - Works well out-of-the-box",
        "   - Minimal hyperparameter tuning required",
        "   - Stable performance across different datasets",
        "",
        "5. EXCELLENT INTERPRETABILITY:",
        "   - Built-in feature importance",
        "   - SHAP values support",
        "   - Model analysis tools included"
    ]
    
    for advantage in advantages:
        print(advantage)


if __name__ == "__main__":
    main()
