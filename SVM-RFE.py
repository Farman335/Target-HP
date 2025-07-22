import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV # Changed from RFE
from sklearn.preprocessing import StandardScaler

def svm_rfe_feature_selection(csv_file_path, target_column, min_features_to_select=1, cv_folds=10, scoring_metric='accuracy'):
    """
    Performs SVM-RFE feature selection with cross-validation (RFECV) on a CSV file.
    RFECV automatically selects the optimal number of features based on cross-validation performance.

    Args:
        csv_file_path (str): The path to the input CSV file.
        target_column (str): The name of the target/dependent variable column.
        min_features_to_select (int): The minimum number of features to consider.
                                      RFECV will select at least this many features.
        cv_folds (int): The number of folds for cross-validation.
        scoring_metric (str): The scoring metric to use for cross-validation (e.g., 'accuracy', 'f1', 'roc_auc').

    Returns:
        list: A list of the names of the selected optimal features.
        pandas.DataFrame: The DataFrame containing only the selected optimal features and the target.
        int: The optimal number of features found by RFECV.
        numpy.ndarray: The cross-validation scores for each number of features.
    """
    try:
        # Load the dataset
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded '{csv_file_path}'. Shape: {df.shape}")

        # Separate features (X) and target (y)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the CSV file.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle potential non-numeric features in X
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) != X.shape[1]:
            print("Warning: Non-numeric columns detected in features. RFE will only work on numeric data.")
            print("Please ensure all feature columns are numeric or preprocess them before running RFE.")
            X = X[numeric_cols] # Filter to only numeric columns for RFE

        if X.empty:
            raise ValueError("No numeric features found in the dataset after filtering.")

        # Scale features - highly recommended for SVMs
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Initialize the SVM estimator
        estimator = SVC(kernel="linear", C=1) # C can be tuned, 1 is a common starting point

        # Initialize RFECV
        # step=1 means removing one feature at each iteration
        # cv=cv_folds specifies the number of cross-validation folds
        # scoring specifies the metric to optimize
        rfe_selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_folds,
            scoring=scoring_metric,
            min_features_to_select=min_features_to_select
        )

        # Fit RFECV to the scaled data
        print(f"Starting RFECV with {cv_folds}-fold cross-validation to find optimal features...")
        rfe_selector.fit(X_scaled_df, y)
        print("RFECV fitting complete.")

        # Get the selected features
        selected_features_mask = rfe_selector.support_
        selected_feature_names = X.columns[selected_features_mask].tolist()

        optimal_num_features = rfe_selector.n_features_
        print(f"\nOptimal number of features found by RFECV: {optimal_num_features}")
        print(f"Selected {len(selected_feature_names)} features:")
        for i, feature_name in enumerate(selected_feature_names):
            print(f"{i+1}. {feature_name}")

        # Create a DataFrame with only the selected features and the target
        df_selected = df[selected_feature_names + [target_column]]

        return selected_feature_names, df_selected, optimal_num_features, rfe_selector.cv_results_['mean_test_score'] # Return mean test scores

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return [], None, None, None
    except ValueError as e:
        print(f"Error: {e}")
        return [], None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], None, None, None

# --- Example Usage ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt # Import here for example usage

    # Create a dummy CSV file for demonstration purposes
    dummy_data = {
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100) * 10,
        'feature_3': np.random.rand(100) * 0.5,
        'irrelevant_1': np.random.rand(100) * 100, # Irrelevant feature
        'feature_4': np.random.rand(100) + 5,
        'irrelevant_2': np.random.randn(100),    # Irrelevant feature
        'feature_5': np.random.rand(100) * 2,
        'target': np.random.randint(0, 2, 100) # Binary target variable
    }
    # Add more features to ensure we have enough to select from
    for i in range(6, 250): # Create 244 more dummy features
        dummy_data[f'feature_{i}'] = np.random.rand(100) * np.random.randint(1, 100)

    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = 'dummy_data_for_rfe.csv'
    dummy_df.to_csv(dummy_csv_path, index=False)
    print(f"Dummy CSV file '{dummy_csv_path}' created for demonstration.")

    # Define your CSV file path and target column
    my_csv_file = dummy_csv_path # Replace with your actual CSV file path
    my_target_column = 'target'

    # Run SVM-RFE with 10-fold CV
    # RFECV will determine the *optimal* number of features.
    # min_features_to_select ensures it considers at least this many features.
    selected_features, df_with_selected_features, optimal_num_features, cv_scores = svm_rfe_feature_selection(
        csv_file_path=my_csv_file,
        target_column=my_target_column,
        min_features_to_select=1, # Start searching from 1 feature
        cv_folds=10,
        scoring_metric='accuracy'
    )

    if selected_features:
        print(f"\nDataFrame with selected optimal features and target (first 5 rows):")
        print(df_with_selected_features.head())
        print(f"\nShape of DataFrame with selected features: {df_with_selected_features.shape}")
        
        # Plotting the CV scores to visualize the optimal number of features
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel(f"Cross validation score ({scoring_metric})")
        # RFECV's cv_results_['mean_test_score'] corresponds to the number of features from min_features_to_select
        # up to the total number of features.
        plt.plot(range(min_features_to_select, len(cv_scores) + min_features_to_select), cv_scores)
        plt.title("RFECV - Optimal Number of Features")
        plt.grid(True)
        plt.show()
    else:
        print("Feature selection could not be completed.")