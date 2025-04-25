import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras import regularizers


def preprocess_data(path):
    df = pd.read_csv(path)
    df['flight_datetime_c'] = pd.to_datetime(df['flight_datetime_c'])

    df['hour'] = df['flight_datetime_c'].dt.hour
    df['dayofweek'] = df['flight_datetime_c'].dt.dayofweek
    df['month'] = df['flight_datetime_c'].dt.month
    df['day'] = df['flight_datetime_c'].dt.day

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    df.drop(columns=['hour', 'month', 'dayofweek', 'day'], inplace=True)
    df = pd.get_dummies(df, columns=["flight_phase"])

    sensor_cols = ['flight_cycle', 'eposition', 'egt_probe_average', 'fuel_flw', 'core_spd', 'zpn12p',
                   'vib_n1_#1_bearing', 'vib_n2_#1_bearing', 'vib_n2_turbine_frame',
                   'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                   'dayofweek_sin', 'dayofweek_cos', 'day_sin', 'day_cos'] + \
        [col for col in df.columns if col.startswith("flight_phase_")]

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(split.split(df, groups=df['eng_number']))
    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()

    split_val_test = GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(split_val_test.split(
        df_temp, groups=df_temp['eng_number']))
    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()

    scaler = MinMaxScaler()
    df_train[sensor_cols] = scaler.fit_transform(df_train[sensor_cols])
    df_val[sensor_cols] = scaler.transform(df_val[sensor_cols])
    df_test[sensor_cols] = scaler.transform(df_test[sensor_cols])

    rul_scaler = MinMaxScaler()
    df_train['RUL'] = rul_scaler.fit_transform(df_train[['RUL']])
    df_val['RUL'] = rul_scaler.transform(df_val[['RUL']])
    df_test['RUL'] = rul_scaler.transform(df_test[['RUL']])

    return df_train, df_val, df_test, sensor_cols, rul_scaler


def create_lstm_windows(data, sensor_cols, window_size=10, stride=1, add_noise=False, noise_std=0.02):
    X, y = [], []
    for engine_id in data['eng_number'].unique():
        engine_df = data[data['eng_number'] ==
                         engine_id].reset_index(drop=True)
        for i in range(0, len(engine_df) - window_size, stride):
            window = engine_df.loc[i:i+window_size-1,
                                   sensor_cols].values.astype(np.float32)
            if add_noise:
                noise = np.random.normal(
                    loc=0.0, scale=noise_std, size=window.shape)
                window += noise
            label = engine_df.loc[i + window_size - 1, 'RUL']
            X.append(window)
            y.append(label)
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape,
                   kernel_regularizer=regularizers.l2(0.0015)))
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=False,
              kernel_regularizer=regularizers.l2(0.003)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu',
              kernel_regularizer=regularizers.l2(0.0015)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=Huber(
        delta=0.1), metrics=[MeanSquaredError()])
    return model


def main(data_path, epochs, batch_size):
    df_train, df_val, df_test, sensor_cols, rul_scaler = preprocess_data(
        data_path)
    X_train, y_train = create_lstm_windows(
        df_train, sensor_cols, add_noise=True)
    X_val, y_val = create_lstm_windows(df_val, sensor_cols)
    X_test, y_test = create_lstm_windows(df_test, sensor_cols)

    mlflow.set_experiment("predictive-maintenance")
    with mlflow.start_run():
        model = build_model((X_train.shape[1], X_train.shape[2]))

        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint_path = "best_lstm2_model.keras"
        checkpoint_cb = ModelCheckpoint(
            checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), callbacks=[early_stop, checkpoint_cb], verbose=1)

        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_plot.png")
        mlflow.log_artifact("training_plot.png")
        plt.close()

        best_model = load_model(checkpoint_path)

        y_pred_val = best_model.predict(X_val).flatten()
        y_pred_val = rul_scaler.inverse_transform(
            y_pred_val.reshape(-1, 1)).flatten()
        y_val = rul_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)

        y_pred_test = best_model.predict(X_test).flatten()
        y_pred_test = rul_scaler.inverse_transform(
            y_pred_test.reshape(-1, 1)).flatten()
        y_test = rul_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

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
                        default='../Data/engines2_data_cleaned_no_outliers.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args.data_path, args.epochs, args.batch_size)
