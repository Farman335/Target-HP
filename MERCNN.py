import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, matthews_corrcoef
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping callback
import os

# Load data from CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Assume all columns except the last are features
    y = data.iloc[:, -1].values  # Assume the last column is the label
    return X, y

# Preprocess data (normalization)
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Build a residual block
def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

# Build a single CNN model with residual connections
def build_residual_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = residual_block(x, 64)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

# Build the multi-headed ensemble model
def build_ensemble_model(input_shape, num_heads=3):
    inputs = Input(shape=input_shape)
    heads = []
    for _ in range(num_heads):
        model = build_residual_cnn(input_shape)
        heads.append(model(inputs))
    outputs = tf.keras.layers.Average()(heads)
    ensemble_model = Model(inputs, outputs)
    ensemble_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return ensemble_model

# Evaluate metrics
def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc, sensitivity, specificity, mcc

# Save model and weights
def save_model_and_weights(model, file_name_prefix):
    model_json = model.to_json()
    with open(f"MRECNN.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f"MRECNN.weights.h5")

# Main function to load data, preprocess, build model, and evaluate using 5-fold cross-validation
def main(file_path):
    X, y = load_data(file_path)
    X = preprocess_data(X)

    # Check original shape of the data
    print(f"Original shape of X: {X.shape}")

    # Reshape X to be compatible with Conv2D input (assuming the data can be reshaped to 2D)
    # Here, we'll try to infer an appropriate shape. Adjust based on your data.
    num_samples = X.shape[0]
    num_features = X.shape[1]

    # Example: Assuming we want to reshape to (num_samples, height, width, channels)
    # We'll need to find height and width such that height * width = num_features
    height = 1
    width = num_features
    channels = 1

    # Adjust height and width if necessary to fit your model's requirements
    # Ensure num_features is a product of height and width
    if num_features % height == 0:
        width = num_features // height

    # Reshape to (-1, height, width, channels)
    X = X.reshape((num_samples, height, width, channels))
    input_shape = (height, width, channels)

    print(f"Reshaped X to: {X.shape}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accs, sens, specs, mccs = [], [], [], []

    # Initialize model outside the cross-validation loop
    ensemble_model = build_ensemble_model(input_shape)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Early stopping callback
        #early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        ensemble_model.fit(X_train, y_train, epochs=20, batch_size=100, verbose=1)

        y_pred = (ensemble_model.predict(X_test) > 0.5).astype(int)
        acc, sensitivity, specificity, mcc = evaluate_metrics(y_test, y_pred)

        accs.append(acc)
        sens.append(sensitivity)
        specs.append(specificity)
        mccs.append(mcc)

    # Save the final model and weights after all folds
    save_model_and_weights(ensemble_model, "MRECNN")

    print(f"Accuracy: {np.mean(accs)}")
    print(f"Sensitivity: {np.mean(sens)}")
    print(f"Specificity: {np.mean(specs)}")
    print(f"MCC: {np.mean(mccs)}")

# Example usage
file_path = 'Split_DDE_angiogenic_train_1000_998.csv'
main(file_path)
