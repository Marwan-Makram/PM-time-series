import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import os

# --------------------- Data Preparation ---------------------------- #


def load_data():
    TRAIN_PATH = os.path.join(
        os.getcwd(), '../Data/engines2_data_cleaned_no_outliers.csv')
    df = pd.read_csv(TRAIN_PATH)
    X = df[[
        'flight_cycle',
        'flight_phase',
        'egt_probe_average',
        'fuel_flw',
        'core_spd',
        'zpn12p',
        'vib_n1_#1_bearing',
        'vib_n2_#1_bearing',
        'vib_n2_turbine_frame'
    ]]
    y = df['RUL']
    return X, y

# --------------------- Preprocessing Setup -------------------------- #


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), ['flight_phase'])
        ],
        remainder='passthrough'
    )

# --------------------- Model Training ------------------------------ #


def train_model():
    mlflow.set_experiment('predictive-maintenance')

    X, y = load_data()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    with mlflow.start_run() as run:
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', create_preprocessor()),
            ('model', RandomForestRegressor(random_state=42))
        ])

        # Hyperparameter tuning
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_

        # Log parameters
        mlflow.log_params(grid_search.best_params_)

        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        mlflow.log_metrics({
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_r2': r2_score(y_val, y_val_pred)
        })

        # Evaluate on test set
        y_test_pred = best_model.predict(X_test)
        mlflow.log_metrics({
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_r2': r2_score(y_test, y_test_pred)
        })

        # Plot and log feature importance
        plot_feature_importance(best_model.named_steps['model'],
                                best_model.named_steps['preprocessor'])

        # Log the complete pipeline
        mlflow.sklearn.log_model(best_model, "model")
        print(f"Run ID: {run.info.run_id}")

# --------------------- Visualization ------------------------------- #


def plot_feature_importance(model, preprocessor):
    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Plot importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i]
               for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importances_rf.png")
    mlflow.log_artifact("feature_importances_rf.png")
    plt.close()


# --------------------- Main Execution ----------------------------- #
if __name__ == "__main__":
    train_model()
