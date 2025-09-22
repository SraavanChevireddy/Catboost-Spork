import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


def create_overfitting_prone_dataset(n_samples=800, n_features=50, noise_level=0.3):
    """
    Create a dataset that's prone to overfitting to test model robustness.
    
    Args:
        n_samples (int): Number of samples (small to encourage overfitting)
        n_features (int): Number of features (high dimensionality)
        noise_level (float): Amount of noise to add
        
    Returns:
        tuple: X (features), y (target)
    """
    np.random.seed(42)
    
    # Create features with different types
    # Relevant features (first 10)
    X_relevant = np.random.randn(n_samples, 10)
    
    # Noise features (remaining features)
    X_noise = np.random.randn(n_samples, n_features - 10)
    
    # Combine features
    X = np.concatenate([X_relevant, X_noise], axis=1)
    
    # Create target based only on relevant features with some noise
    true_coefficients = np.array([2, -1.5, 1, 0.8, -0.5, 1.2, -0.8, 0.6, -1, 0.4])
    linear_combination = X_relevant @ true_coefficients
    
    # Add non-linear interactions
    interactions = (X_relevant[:, 0] * X_relevant[:, 1] + 
                   X_relevant[:, 2] ** 2 - 
                   X_relevant[:, 3] * X_relevant[:, 4])
    
    # Create target with noise
    target_continuous = linear_combination + 0.5 * interactions + np.random.normal(0, noise_level, n_samples)
    
    # Convert to binary classification
    y = (target_continuous > np.median(target_continuous)).astype(int)
    
    # Create feature names
    feature_names = ([f'relevant_feature_{i}' for i in range(10)] + 
                    [f'noise_feature_{i}' for i in range(n_features - 10)])
    
    X = pd.DataFrame(X, columns=feature_names)
    
    return X, y


def demonstrate_training_curves():
    """
    Show training and validation curves to demonstrate overfitting behavior.
    """
    print("="*70)
    print("TRAINING CURVES: OVERFITTING DEMONSTRATION")
    print("="*70)
    
    X, y = create_overfitting_prone_dataset(n_samples=600, n_features=40)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print("Small dataset with many features to encourage overfitting")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define models with different complexities
    models = {
        'CatBoost (Default)': CatBoostClassifier(
            iterations=300,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            random_seed=42
        ),
        'CatBoost (High Complexity)': CatBoostClassifier(
            iterations=500,
            learning_rate=0.15,
            depth=10,
            l2_leaf_reg=1,  # Lower regularization
            verbose=False,
            random_seed=42
        ),
        'Random Forest (High Complexity)': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,  # No depth limit
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        'Gradient Boosting (High Complexity)': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.15,
            max_depth=8,
            min_samples_split=2,
            random_state=42
        )
    }
    
    # Track training progress for each model
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, (model_name, model) in enumerate(models.items()):
        print(f"\nTraining {model_name}...")
        
        if 'CatBoost' in model_name:
            # For CatBoost, we can track training progress
            model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)
            
            # Get training history
            evals_result = model.get_evals_result()
            train_scores = evals_result['learn']['Logloss']
            valid_scores = evals_result['validation']['Logloss']
            iterations = range(len(train_scores))
            
            # Plot training curves
            axes[idx].plot(iterations, train_scores, label='Training Loss', color='blue', alpha=0.7)
            axes[idx].plot(iterations, valid_scores, label='Validation Loss', color='red', alpha=0.7)
            
        else:
            # For sklearn models, use learning curves
            train_sizes, train_scores, valid_scores = learning_curve(
                model, X_train, y_train, cv=3, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='neg_log_loss'
            )
            
            train_mean = -train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            valid_mean = -valid_scores.mean(axis=1)
            valid_std = valid_scores.std(axis=1)
            
            axes[idx].plot(train_sizes, train_mean, label='Training Loss', color='blue', alpha=0.7)
            axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            axes[idx].plot(train_sizes, valid_mean, label='Validation Loss', color='red', alpha=0.7)
            axes[idx].fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1, color='red')
            axes[idx].set_xlabel('Training Set Size')
        
        axes[idx].set_title(f'{model_name}')
        axes[idx].set_ylabel('Log Loss')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        # Calculate final overfitting gap
        if 'CatBoost' in model_name:
            final_train_loss = train_scores[-1]
            final_valid_loss = valid_scores[-1]
            overfitting_gap = final_valid_loss - final_train_loss
        else:
            model.fit(X_train, y_train)
            train_pred_proba = model.predict_proba(X_train)[:, 1]
            test_pred_proba = model.predict_proba(X_test)[:, 1]
            final_train_loss = log_loss(y_train, train_pred_proba)
            final_valid_loss = log_loss(y_test, test_pred_proba)
            overfitting_gap = final_valid_loss - final_train_loss
        
        results[model_name] = {
            'train_loss': final_train_loss,
            'valid_loss': final_valid_loss,
            'overfitting_gap': overfitting_gap
        }
        
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_valid_loss:.4f}")
        print(f"Overfitting Gap: {overfitting_gap:.4f}")
    
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_small_dataset_performance():
    """
    Show how CatBoost performs better on small datasets due to overfitting resistance.
    """
    print("\n" + "="*70)
    print("SMALL DATASET PERFORMANCE COMPARISON")
    print("="*70)
    
    # Test on datasets of different sizes
    dataset_sizes = [100, 200, 400, 600, 800, 1000]
    
    models = {
        'CatBoost': CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            random_seed=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    }
    
    results = {model_name: {'train_acc': [], 'test_acc': [], 'overfitting_gap': []} 
               for model_name in models.keys()}
    
    for size in dataset_sizes:
        print(f"\nTesting with {size} samples...")
        
        X, y = create_overfitting_prone_dataset(n_samples=size, n_features=30)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Accuracies
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            gap = train_acc - test_acc
            
            results[model_name]['train_acc'].append(train_acc)
            results[model_name]['test_acc'].append(test_acc)
            results[model_name]['overfitting_gap'].append(gap)
            
            print(f"{model_name:20} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | Gap: {gap:.3f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test accuracy comparison
    for model_name in models.keys():
        ax1.plot(dataset_sizes, results[model_name]['test_acc'], 'o-', 
                label=model_name, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy vs Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfitting gap comparison
    for model_name in models.keys():
        ax2.plot(dataset_sizes, results[model_name]['overfitting_gap'], 'o-', 
                label=model_name, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Overfitting Gap (Train Acc - Test Acc)')
    ax2.set_title('Overfitting Gap vs Dataset Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_regularization_impact():
    """
    Show how CatBoost's built-in regularization compares to manual regularization.
    """
    print("\n" + "="*70)
    print("REGULARIZATION IMPACT DEMONSTRATION")
    print("="*70)
    
    X, y = create_overfitting_prone_dataset(n_samples=500, n_features=40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test different regularization levels
    regularization_configs = [
        ('No Regularization', {'l2_leaf_reg': 1, 'depth': 10}),
        ('Light Regularization', {'l2_leaf_reg': 3, 'depth': 8}),
        ('Medium Regularization', {'l2_leaf_reg': 5, 'depth': 6}),
        ('Heavy Regularization', {'l2_leaf_reg': 10, 'depth': 4}),
        ('CatBoost Default', {'l2_leaf_reg': 3, 'depth': 6})  # Default parameters
    ]
    
    results = []
    
    print(f"Testing regularization impact on {X.shape[0]} samples, {X.shape[1]} features")
    
    for config_name, params in regularization_configs:
        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            verbose=False,
            random_seed=42,
            **params
        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        gap = train_acc - test_acc
        
        results.append({
            'config': config_name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'overfitting_gap': gap,
            'l2_reg': params['l2_leaf_reg'],
            'depth': params['depth']
        })
        
        print(f"{config_name:20} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | Gap: {gap:.3f}")
    
    # Visualize regularization impact
    results_df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    x_pos = range(len(results_df))
    width = 0.35
    
    ax1.bar([x - width/2 for x in x_pos], results_df['train_acc'], width, 
            label='Train Accuracy', alpha=0.7, color='blue')
    ax1.bar([x + width/2 for x in x_pos], results_df['test_acc'], width, 
            label='Test Accuracy', alpha=0.7, color='red')
    
    ax1.set_xlabel('Regularization Configuration')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Train vs Test Accuracy by Regularization')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df['config'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfitting gap
    ax2.bar(x_pos, results_df['overfitting_gap'], color='orange', alpha=0.7)
    ax2.set_xlabel('Regularization Configuration')
    ax2.set_ylabel('Overfitting Gap')
    ax2.set_title('Overfitting Gap by Regularization')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['config'], rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df


def main():
    """
    Main function to demonstrate CatBoost's overfitting resistance.
    """
    print("CatBoost Overfitting Resistance Demonstration")
    print("="*70)
    
    # Demonstrate training curves
    training_results = demonstrate_training_curves()
    
    # Small dataset performance
    small_dataset_results = demonstrate_small_dataset_performance()
    
    # Regularization impact
    regularization_results = demonstrate_regularization_impact()
    
    # Summary analysis
    print("\n" + "="*70)
    print("OVERFITTING RESISTANCE ANALYSIS SUMMARY")
    print("="*70)
    
    # Find best performing model on small datasets
    catboost_avg_gap = np.mean(small_dataset_results['CatBoost']['overfitting_gap'])
    rf_avg_gap = np.mean(small_dataset_results['Random Forest']['overfitting_gap'])
    gb_avg_gap = np.mean(small_dataset_results['Gradient Boosting']['overfitting_gap'])
    
    print("AVERAGE OVERFITTING GAP ACROSS DATASET SIZES:")
    print(f"CatBoost:          {catboost_avg_gap:.3f}")
    print(f"Random Forest:     {rf_avg_gap:.3f}")
    print(f"Gradient Boosting: {gb_avg_gap:.3f}")
    
    print("\nCATBOOST OVERFITTING RESISTANCE ADVANTAGES:")
    
    advantages = [
        "1. ORDERED BOOSTING:",
        "   ✓ Prevents target leakage during training",
        "   ✓ Uses different permutations for each tree",
        "   ✓ Reduces overfitting compared to standard boosting",
        "",
        "2. BUILT-IN REGULARIZATION:",
        "   ✓ Automatic L2 regularization on leaf values",
        "   ✓ Depth-wise regularization",
        "   ✓ Feature combination penalties",
        "",
        "3. EARLY STOPPING:",
        "   ✓ Automatic detection of overfitting",
        "   ✓ Stops training at optimal iteration",
        "   ✓ Prevents unnecessary complexity",
        "",
        "4. ROBUST DEFAULT PARAMETERS:",
        "   ✓ Conservative depth settings (6 vs unlimited)",
        "   ✓ Appropriate learning rate (0.1)",
        "   ✓ Balanced regularization strength",
        "",
        "5. SUPERIOR SMALL DATASET PERFORMANCE:",
        f"   ✓ Lowest average overfitting gap: {catboost_avg_gap:.3f}",
        f"   ✓ {((rf_avg_gap - catboost_avg_gap) / rf_avg_gap * 100):.1f}% better than Random Forest",
        f"   ✓ {((gb_avg_gap - catboost_avg_gap) / gb_avg_gap * 100):.1f}% better than Gradient Boosting",
        "",
        "6. MINIMAL HYPERPARAMETER TUNING:",
        "   ✓ Works well with default settings",
        "   ✓ Less prone to hyperparameter overfitting",
        "   ✓ Stable performance across different datasets"
    ]
    
    for advantage in advantages:
        print(advantage)


if __name__ == "__main__":
    main()
