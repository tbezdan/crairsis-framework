import pandas as pd
import os
from utils.config import output_path


def format_best_models(output_path):
    optimized_df = pd.read_csv(
        os.path.join(output_path, "results", "optimized_models_results_cv.csv")
    )

    # Find the best model for each combination of target and filename based on R^2
    best_models_df = optimized_df.loc[
        optimized_df.groupby(["Target", "Covid", "Filename"])["R2"].idxmax()
    ]

    # Create a new DataFrame with the desired structure
    best_models_structured_df = best_models_df[
        ["ML Model", "R2", "Filename", "Metaheuristic"]
    ].copy()

    # Rename columns to match the desired output
    best_models_structured_df.rename(
        columns={
            "ML Model": "ml_model",
            "R2": "Value",
            "Metaheuristic": "mh_algo",
            "Target": "target",
            "Covid": "covid",
            "Filename": "site",
        },
        inplace=True,
    )

    # Add constant columns
    best_models_structured_df["Name"] = "R^2"
    best_models_structured_df["site"] = best_models_df["Filename"]
    best_models_structured_df["covid"] = best_models_df["Covid"]
    best_models_structured_df["target"] = best_models_df["Target"]

    # Reorder columns to match the desired format
    structured_df = best_models_structured_df[
        ["Name", "Value", "site", "covid", "target", "ml_model", "mh_algo"]
    ]

    # Save the structured best models to CSV
    structured_df.to_csv(os.path.join(output_path, "best_models.csv"), index=False)


def format_detailed_metrics(output_path):
    optimized_df = pd.read_csv(
        os.path.join(output_path, "results", "optimized_models_results.csv")
    )

    detailed_metrics = []
    for index, row in optimized_df.iterrows():
        for metric_name in [
            "MAE",
            "MSE",
            "RMSE",
            "MAPE",
            "Explained Variance",
            "Max Error",
            "R2",
        ]:
            detailed_metrics.append(
                {
                    "Name": metric_name,
                    "Value": row[metric_name],
                    "Site": row["Filename"],
                    "Covid": row["Covid"],
                    "Target": row["Target"],
                    "ML Model": row["ML Model"],
                    "Metaheuristic": row["Metaheuristic"],
                }
            )

    # Convert the detailed metrics list to a DataFrame
    detailed_metrics_df = pd.DataFrame(detailed_metrics)

    detailed_metrics_df.rename(
        columns={
            "ML Model": "ml_model",
            "Metaheuristic": "mh_algo",
            "Target": "target",
            "Covid": "covid",
            "Site": "site",
        },
        inplace=True,
    )

    # Save the detailed metrics DataFrame to a CSV file
    detailed_metrics_df.to_csv(
        os.path.join(output_path, "best_models_and_metrics.csv"),
        index=False,
    )
