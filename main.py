from src.data_processing import load_and_preprocess_data
from src.model_training import train_models
from src.hyperparameter_tuning import tune_random_forest
from src.visualization import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_line,
)


def main():
    data_path = "data/raw/daily-bike-share.csv"
    processed_data_path = "data/processed"

    X_train, X_test, Y_train, Y_test = load_and_preprocess_data(
        data_path, processed_data_path
    )
    best_model_name = train_models(X_train, Y_train, X_test, Y_test)

    if best_model_name == "Random Forest":
        print("\nPerforming Hyperparameter Tuning for Random Forest...")
        best_params = tune_random_forest(X_train, Y_train)

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score

        print("Training Random Forest with Best Parameters...")
        rf_model = RandomForestRegressor(**best_params, random_state=0)
        rf_model.fit(X_train, Y_train)

        Y_pred = rf_model.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)
        print(f"\nNew R2 Score after Hyperparameter Tuning: {r2:.4f}")

        # Visualization
        plot_actual_vs_predicted(Y_test, Y_pred)
        plot_residuals(Y_test, Y_pred)
        plot_feature_importance(rf_model.feature_importances_, X_train.columns)
        plot_line(Y_test, Y_pred)


if __name__ == "__main__":
    main()
