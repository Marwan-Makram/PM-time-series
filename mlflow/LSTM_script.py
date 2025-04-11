import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Initialize MLflow
mlflow.set_experiment("predictive-maintenance")
mlflow.tensorflow.autolog()


def create_sequences_all(df, features, target, window_size):
    """Create sequences for all engines in the dataframe."""
    X, y = [], []
    for eng in df['eng_number'].unique():
        engine_data = df[df['eng_number'] == eng]
        # Ensure engine has enough data points
        if len(engine_data) <= window_size:
            continue
        for i in range(len(engine_data) - window_size):
            X.append(engine_data[features].iloc[i:i +
                     window_size].values.astype(np.float32))
            y.append(engine_data[target].iloc[i + window_size])
    return np.array(X), np.array(y)


def build_model(window_size, n_features):
    """Build and compile the Conv1D-LSTM model."""
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(window_size, n_features)),
        tf.keras.layers.Conv1D(
            64, 3, strides=1, activation="relu", padding='causal'),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main():
    # Parameters
    WINDOW_SIZE = 10
    BATCH_SIZE = 64
    EPOCHS = 100

    with mlflow.start_run():
        # Load and preprocess data
        df = pd.read_csv('../Data/engines2_data_cleaned_no_outliers.csv')

        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True)

        df = pd.get_dummies(df, columns=['flight_phase'], drop_first=True)

        # Define target and features
        target = 'RUL'
        non_features = ['eng_number', target]
        features = [col for col in df.columns
                    if col not in non_features and pd.api.types.is_numeric_dtype(df[col])]

        # Split engines into train/val/test
        engines = df['eng_number'].unique()
        np.random.seed(42)
        np.random.shuffle(engines)
        n_engines = len(engines)
        train_engines = engines[:int(0.8 * n_engines)]
        val_engines = engines[int(0.8 * n_engines): int(0.9 * n_engines)]
        test_engines = engines[int(0.9 * n_engines):]

        # Split data into respective dataframes
        train_df = df[df['eng_number'].isin(train_engines)].copy()
        val_df = df[df['eng_number'].isin(val_engines)].copy()
        test_df = df[df['eng_number'].isin(test_engines)].copy()

        # Normalization
        scaler = MinMaxScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        val_df[features] = scaler.transform(val_df[features])
        test_df[features] = scaler.transform(test_df[features])

        # Generate sequences
        X_train, y_train = create_sequences_all(
            train_df, features, target, WINDOW_SIZE)
        X_val, y_val = create_sequences_all(
            val_df, features, target, WINDOW_SIZE)
        X_test, y_test = create_sequences_all(
            test_df, features, target, WINDOW_SIZE)

        # Shuffle training data
        shuffle_idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]

        # Build model
        model = build_model(WINDOW_SIZE, len(features))

        # Log parameters
        mlflow.log_params({
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "features": features,
            "model_type": "Conv1D-LSTM",
            "n_train_engines": len(train_engines),
            "n_val_engines": len(val_engines),
            "n_test_engines": len(test_engines)
        })

        # Training callbacks
        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1
        )

        # Evaluate on test set
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mlflow.log_metrics({"test_loss": test_loss, "test_mae": mae})

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Training history
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_title('Training History')
        ax1.legend()

        # Prediction samples
        sample_indices = np.arange(50)
        y_true_sample = y_test[:50]
        y_pred_sample = y_pred[:50].flatten()
        ax2.plot(sample_indices, y_true_sample, 'o-', label='True RUL')
        ax2.plot(sample_indices, y_pred_sample, 'x-', label='Predicted RUL')
        ax2.set_title('True vs. Predicted RUL (First 50 Test Samples)')
        ax2.legend()

        mlflow.log_figure(fig, "training_results.png")
        plt.close()

        # Save model
        mlflow.tensorflow.log_model(model, "model")


if __name__ == "__main__":
    main()
