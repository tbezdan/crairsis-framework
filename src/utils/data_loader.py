import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.config import datasets_path
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(
    df,
    target_column,
    datetime_col,
    test_size=0.2,
    random_state=42,
    filename=None,
    filter_column=None,
    filter_value=None,
    target_name=None,
    split_data=True,
    task_type=None,
):
    """
    Load, preprocess the dataset, optionally split it, add 'Usage' and 'id' columns, and save the modified dataframe.
    """

    if task_type == "classification":
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column].astype(str))

    df_copy = df.copy()

    if split_data:
        X = df_copy.drop([target_column], axis=1)
        y = df_copy[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if task_type == "classification" else None,
        )

        X_train["usage"] = "train"
        X_test["usage"] = "test"
        combined_df = pd.concat([X_train, X_test])
        combined_df[target_column] = pd.concat([y_train, y_test])

        combined_df.to_csv(
            os.path.join(
                datasets_path,
                f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target_name}.csv",
            ),
            index=False,
        )

        X_train = X_train.drop(columns=["usage"], errors="ignore")
        X_test = X_test.drop(columns=["usage"], errors="ignore")

        if datetime_col in X_train.columns:
            X_train = X_train.drop(columns=[datetime_col], errors="ignore")
            X_test = X_test.drop(columns=[datetime_col], errors="ignore")

        if "id" in X_train.columns:
            X_train = X_train.drop(columns=["id"], errors="ignore")
            X_test = X_test.drop(columns=["id"], errors="ignore")

        return X_train, X_test, y_train, y_test
    else:

        if datetime_col in df_copy.columns:
            df_copy = df_copy.drop(columns=[datetime_col], errors="ignore")
        if "id" in df_copy.columns:
            df_copy = df_copy.drop(columns=["id"], errors="ignore")
        return df_copy
