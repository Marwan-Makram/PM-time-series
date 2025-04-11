import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

## --------------------- Data Preparation ---------------------------- ##

# Read the Dataset
TRAIN_PATH = os.path.join(
    os.getcwd(), '../Data/engines2_data_cleaned_no_outliers.csv')
df = pd.read_csv(TRAIN_PATH)

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

# Split to train, validation, and test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

## --------------------- Hyperparameter Tuning ---------------------------- ##

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

# Random Forest regression model
rf = RandomForestRegressor(random_state=42)

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# --------------------- Start MLflow Run ---------------------------- ##
mlflow.set_experiment(f'predictive-maintenance')

with mlflow.start_run() as run:
    # Log the best hyperparameters to MLflow
    mlflow.log_params(grid_search.best_params_)
    print("Best Parameters:", grid_search.best_params_)

    ## --------------------- Model Evaluation ---------------------------- ##

    def evaluate_model(model, X, y, dataset_name):
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        print(f"{dataset_name} MAE: {mae}, RMSE: {rmse}, R^2: {r2}")

        return mae, mse, rmse, r2

    # Evaluate on Validation Set
    mae_val, mse_val, rmse_val, r2_val = evaluate_model(
        best_rf, X_val, y_val, 'Validation')
    mlflow.log_metrics({
        'val_mae': mae_val,
        'val_rmse': rmse_val,
        'val_r2': r2_val
    })

    # Evaluate on Test Set
    mae_test, mse_test, rmse_test, r2_test = evaluate_model(
        best_rf, X_test, y_test, 'Test')
    mlflow.log_metrics({
        'test_mae': mae_test,
        'test_rmse': rmse_test,
        'test_r2': r2_test
    })

    ## --------------------- Feature Importance ---------------------------- ##

    importances = best_rf.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()

    # Save the plot and log it to MLflow
    plt.savefig('feature_importances.png')
    plt.show()
    mlflow.log_artifact('feature_importances.png')

    ## --------------------- Model Logging ---------------------------- ##

    # Log the model itself to MLflow
    mlflow.sklearn.log_model(best_rf, 'model')
