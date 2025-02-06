import pandas as pd
import os


def load_and_preprocess_data(data_path, save_path):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    df = pd.get_dummies(
        df,
        columns=[
            "season",
            "mnth",
            "weekday",
            "weathersit",
            "yr",
            "holiday",
            "workingday",
        ],
        drop_first=True,
    )
    X = df.drop(columns=["instant", "dteday", "rentals"])
    Y = df["rentals"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0
    )

    os.makedirs(save_path, exist_ok=True)
    processed_file_path = os.path.join(save_path, "processed_bike_data.csv")
    df.to_csv(processed_file_path, index=False)

    print(f"Processed dataset saved to: {processed_file_path}")
    return X_train, X_test, Y_train, Y_test
