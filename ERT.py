import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier # Changed from xgboost.XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def initialize_and_train_ert(X_train, y_train, X_test, y_test): # Function name changed to reflect ERT
    """
    Initializes an ExtraTreesClassifier (ERT) with specified optimal hyperparameters
    and trains it on the provided data.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target variable.
        X_test (pd.DataFrame or np.ndarray): Testing features.
        y_test (pd.Series or np.ndarray): Testing target variable.

    Returns:
        tuple: A tuple containing:
            - model (ExtraTreesClassifier): The trained ERT model.
            - y_pred (np.ndarray): Predicted labels for the test set.
            - y_proba (np.ndarray): Predicted probabilities for the positive class on the test set.
    """
    # Define the optimal hyperparameters for ERT as identified by grid search
    optimal_params = {
        'n_estimators': 300,
        'max_depth': None, # Nodes expanded until all leaves are pure
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'max_features': 'auto', # Number of features considered for splitting at each node
        'bootstrap': False,
        'criterion': 'gini' # Splitting criteria
    }

    print("Initializing ExtraTreesClassifier (ERT) with optimal hyperparameters:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value}")

    # Initialize the ExtraTreesClassifier with the optimal parameters
    model = ExtraTreesClassifier(**optimal_params) # Changed to ExtraTreesClassifier

    # Train the model
    print("\nTraining ExtraTreesClassifier (ERT) model...")
    model.fit(X_train, y_train)
    print("ExtraTreesClassifier (ERT) model training complete.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

    return model, y_pred, y_proba

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create a dummy dataset for demonstration
    # In a real scenario, you would load your actual dataset (e.g., from a CSV)
    from sklearn.datasets import make_classification

    print("Creating a dummy classification dataset for demonstration...")
    X, y = make_classification(
        n_samples=1000,      # Number of samples
        n_features=20,       # Number of features
        n_informative=10,    # Number of informative features
        n_redundant=5,       # Number of redundant features
        n_classes=2,         # Binary classification
        random_state=42      # For reproducibility
    )
    
    # Convert to DataFrame for easier handling, especially if you have feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    print(f"Dataset split into training (X_train: {X_train.shape}, y_train: {y_train.shape}) "
          f"and testing (X_test: {X_test.shape}, y_test: {y_test.shape}).")

    # 2. Call the function to initialize and train the ERT model
    trained_model, predictions, probabilities = initialize_and_train_ert( # Function call changed
        X_train, y_train, X_test, y_test
    )

    # 3. Evaluate the model's performance
    print("\n--- Model Evaluation ---")
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print(f"\nFirst 10 predicted probabilities: {probabilities[:10]}")
    print(f"First 10 predicted labels: {predictions[:10]}")