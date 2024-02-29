import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config import datasets_path


def load_and_preprocess_data(
    df,
    target_column,
    output_dir,
    test_size=0.2,
    random_state=42,
    site=None,
    covid=None,
    target_name=None,
    split_data=True,
):
    """
    Load, preprocess the dataset, optionally split it, add 'Usage' and 'id' columns, and save the modified dataframe.
    """

    df = df[df[target_column] != 0]
    # Create a copy of the DataFrame to work with
    df_copy = df.copy()

    # Split the dataset into training and testing sets
    if split_data:
        X = df_copy.drop([target_column], axis=1)
        y = df_copy[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Combine the training and testing sets along with their labels for saving
        X_train["Usage"] = "Train"
        X_test["Usage"] = "Test"
        combined_df = pd.concat([X_train, X_test])
        combined_df[target_column] = pd.concat([y_train, y_test])

        # Save the combined dataset
        combined_df.to_csv(
            os.path.join(
                datasets_path,
                f"site_{site}_covid_{covid}_target_{target_name}.csv",
            ),
            index=False,
        )

        # Return the split data
        X_train = X_train.drop(columns=["Usage", "Datetime", "id"])
        X_test = X_test.drop(columns=["Usage", "Datetime", "id"])

        return X_train, X_test, y_train, y_test
    else:
        # If no split is needed, just return the preprocessed data without 'Datetime' and 'id'
        return df_copy.drop(["Datetime", "id"], axis=1)
