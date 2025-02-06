from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor


def tune_random_forest(X, Y):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    model = RandomForestRegressor()
    grid_search = GridSearchCV(
        model, param_grid, cv=kf, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X, Y)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    return grid_search.best_params_
