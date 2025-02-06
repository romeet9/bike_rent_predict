import matplotlib.pyplot as plt


def plot_actual_vs_predicted(Y_test, Y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_test, Y_pred, alpha=0.5, color="blue")
    plt.plot(
        [min(Y_test), max(Y_test)],
        [min(Y_test), max(Y_test)],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Actual Rentals")
    plt.ylabel("Predicted Rentals")
    plt.title("Actual vs Predicted Bike Rentals")
    plt.show()


def plot_residuals(Y_test, Y_pred):
    residuals = Y_test - Y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_pred, residuals, alpha=0.5, color="green")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Rentals")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot for Predictions")
    plt.show()


def plot_feature_importance(feature_importances, feature_names):
    sorted_indices = feature_importances.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(
        range(len(sorted_indices)), feature_importances[sorted_indices], align="center"
    )
    plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.show()


def plot_line(Y_test, Y_pred):
    plt.plot(Y_test.values, label="Actual Values")
    plt.plot(Y_pred, label="Predictions")
    plt.xlabel("Time")
    plt.ylabel("Bike Rentals")
    plt.legend()
    plt.show()
