import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError


def preprocess_data(path):
    df = pd.read_csv(path)
    df['flight_datetime_c'] = pd.to_datetime(
        df['flight_datetime'], format='%d-%m-%y %H:%M', dayfirst=True)

    df['hour'] = df['flight_datetime_c'].dt.hour
    df['month'] = df['flight_datetime_c'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df.drop(columns=['hour', 'month'], inplace=True)
    df = pd.get_dummies(df, columns=["flight_phase"])

    sensor_cols = ['eposition', 'egt_probe_average', 'fuel_flw', 'core_spd', 'zpn12p',
                   'vib_n1_1_bearing', 'vib_n2_1_bearing', 'vib_n2_turbine_frame',
                   'hour_sin', 'hour_cos', 'month_sin', 'month_cos'] + \
        [col for col in df.columns if col.startswith("flight_phase_")]

    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df, sensor_cols


def create_lstm_windows(data, sensor_cols, window_size=15):
    X, y = [], []
    for engine_id in data['eng_number'].unique():
        engine_df = data[data['eng_number'] ==
                         engine_id].reset_index(drop=True)
        for i in range(len(engine_df) - window_size):
            window = engine_df.loc[i:i+window_size-1,
                                   sensor_cols].values.astype(np.float32)
            label = engine_df.loc[i + window_size - 1, 'RUL']
            X.append(window)
            y.append(label)
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=MeanAbsoluteError(),
                  metrics=[MeanSquaredError()])
    return model


def main(data_path, epochs, batch_size):
    df, sensor_cols = preprocess_data(data_path)
    X, y = create_lstm_windows(df, sensor_cols)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    mlflow.set_experiment("predictive-maintenance")
    with mlflow.start_run():
        model = build_model(input_shape=(X.shape[1], X.shape[2]))

        early_stop = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True)
        checkpoint_path = "best_lstm_model.keras"
        checkpoint_cb = ModelCheckpoint(
            checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), callbacks=[early_stop, checkpoint_cb], verbose=1)

        # Plot and log training history
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MAE Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_plot.png")
        mlflow.log_artifact("training_plot.png")
        plt.close()

        best_model = load_model(checkpoint_path)

        # Validation
        y_pred_val = best_model.predict(X_val).flatten()
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)

        # Test
        y_pred_test = best_model.predict(X_test).flatten()
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        mlflow.log_params({'epochs': epochs, 'batch_size': batch_size})
        mlflow.log_metrics({
            'val_mae': val_mae, 'val_r2': val_r2,
            'test_mae': test_mae, 'test_r2': test_r2
        })

        mlflow.keras.log_model(best_model, artifact_path="models")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='../Data/engines2_data_cleaned_no_outliers_lstm.csv')
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args.data_path, args.epochs, args.batch_size)
