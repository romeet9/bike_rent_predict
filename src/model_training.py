from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def train_models(X_train, Y_train, X_test, Y_test):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=0),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=0),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        results.append({"Model": name, "MAE": mean_absolute_error(Y_test, Y_pred),
                        "MSE": mean_squared_error(Y_test, Y_pred), "R2 Score": r2_score(Y_test, Y_pred)})
    
    results_df = pd.DataFrame(results)
    print("\nModel Evaluation Results:")
    print(results_df)
    
    best_model_name = results_df.loc[results_df["R2 Score"].idxmax()]['Model']
    print(f"\nBest Model: {best_model_name}")
    return best_model_name
