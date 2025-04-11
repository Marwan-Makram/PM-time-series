import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor

# --------------------- Data Preparation ---------------------------- ##

# Load the data
df = pd.read_csv('../Data/engines2_data_cleaned_no_outliers.csv')

# Ensure df is a DataFrame
if isinstance(df, pd.DataFrame):
    # Select relevant columns
    X = df[['flight_cycle', 'flight_phase', 'egt_probe_average', 'fuel_flw', 'core_spd',
            'zpn12p', 'vib_n1_#1_bearing', 'vib_n2_#1_bearing', 'vib_n2_turbine_frame']].copy()
else:
    raise TypeError(
        "The variable 'df' is not a DataFrame. Please ensure it is properly defined.")
Y = df['RUL'].copy()

# Get dummy for flight_phase
X = pd.get_dummies(X, columns=['flight_phase'], drop_first=True)

# Initial 80/20 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y, test_size=0.2, random_state=42)
# Further split temp into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# --------------------- Start MLflow Run ---------------------------- ##
mlflow.set_experiment('predictive-maintenance')

with mlflow.start_run() as run:

    # --------------------- Model Training and Hyperparameter Tuning ---------------------------- ##

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {'n_estimators': [100, 200, 300]}
    xgb_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5,
                               scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    best_xgb = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    print("Best Parameters:", grid_search.best_params_)

    # --------------------- Model Evaluation ---------------------------- ##

    # Evaluate on validation set
    y_val_pred = best_xgb.predict(X_val_scaled)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_val, y_val_pred)
    print(
        f"Validation MAE: {mae_val}, Validation RMSE: {rmse_val}, Validation R^2: {r2_val}")

    mlflow.log_metrics({
        'val_mae': mae_val,
        'val_rmse': rmse_val,
        'val_r2': r2_val
    })

    # Evaluate on test set
    y_test_pred = best_xgb.predict(X_test_scaled)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"Test MAE: {mae_test}, Test RMSE: {rmse_test}, Test R^2: {r2_test}")

    mlflow.log_metrics({
        'test_mae': mae_test,
        'test_rmse': rmse_test,
        'test_r2': r2_test
    })

    # --------------------- Feature Importance ---------------------------- ##

    # Get feature importances and plot them
    importances = best_xgb.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()

    # Save the plot and log it as an artifact
    plt.savefig('feature_importances_xgb.png')
    plt.show()
    mlflow.log_artifact('feature_importances_xgb.png')

    # --------------------- Model Logging ---------------------------- ##

    # Log the model itself to MLflow
    mlflow.sklearn.log_model(best_xgb, 'model')
