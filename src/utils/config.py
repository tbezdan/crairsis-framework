from mealpy import FloatVar, IntegerVar
import os

# ROOT = "/Users/timea/Documents/Projekti/craAIRsis/Covid BG"
ROOT = "/Users/timea/Documents/Projekti/craAIRsis/TEST"

original_datasets_path = os.path.join(ROOT, "original datasets")


models_path = os.path.join(ROOT, "models")
output_path = os.path.join(ROOT, "data")
json_path = os.path.join(ROOT, "data", "json_options")
shap_folder = os.path.join(ROOT, "data", "shap_local_relative_normalized")
shap_clusters_folder = os.path.join(ROOT, "data", "shap_clusters")
sage_folder = os.path.join(ROOT, "data", "sage")
interactions_folder = os.path.join(ROOT, "data", "interactions")
interactions_feature_folder = os.path.join(
    ROOT, "data", "interactions", "interactions_feature"
)
destination_folder = os.path.join(ROOT, "data", "actual_predicted")
original_data_id_folder = os.path.join(ROOT, "data", "EEA_city_full_ID")
results_folder = os.path.join(ROOT, "data", "results")
datasets_path = os.path.join(ROOT, "data", "datasets")


paths_to_check = [
    models_path,
    output_path,
    json_path,
    shap_folder,
    shap_clusters_folder,
    sage_folder,
    interactions_folder,
    interactions_feature_folder,
    destination_folder,
    original_data_id_folder,
    results_folder,
    datasets_path,
]


best_models_path = os.path.join(ROOT, "data", "best_models.csv")

catboost_hyperparams = [
    IntegerVar(3, 10, name="depth"),
    FloatVar(0.001, 0.3, name="learning_rate"),
    IntegerVar(100, 1000, name="iterations"),
    FloatVar(1, 10, name="l2_leaf_reg"),
    IntegerVar(1, 255, name="border_count"),
    FloatVar(0.01, 1, name="bagging_temperature"),
    FloatVar(1e-9, 10, name="random_strength"),
]

adaboost_hyperparams = [
    IntegerVar(10, 100, name="n_estimators"),
    FloatVar(0.01, 1.0, name="learning_rate"),
]

lgbm_hyperparams = [
    IntegerVar(10, 300, name="num_leaves"),
    IntegerVar(10, 100, name="max_depth"),
    FloatVar(0.001, 0.2, name="learning_rate"),
    IntegerVar(100, 2000, name="n_estimators"),
    FloatVar(0.6, 1.0, name="subsample"),
    FloatVar(0.6, 1.0, name="colsample_bytree"),
    FloatVar(0, 1, name="reg_alpha"),
    FloatVar(0, 1, name="reg_lambda"),
]


xgboost_hyperparams = [
    IntegerVar(3, 10, name="max_depth"),
    FloatVar(0.001, 0.3, name="learning_rate"),
    IntegerVar(100, 1000, name="n_estimators"),
    FloatVar(0.1, 1.0, name="subsample"),
    FloatVar(0.1, 1.0, name="colsample_bytree"),
    FloatVar(0, 10, name="reg_alpha"),
    FloatVar(0, 10, name="reg_lambda"),
    FloatVar(0, 10, name="gamma"),
    FloatVar(0, 10, name="min_child_weight"),
]

extratrees_hyperparams = [
    IntegerVar(10, 1000, name="n_estimators"),
    IntegerVar(1, 100, name="max_depth"),
    FloatVar(0.1, 1.0, name="max_features"),
    IntegerVar(2, 10, name="min_samples_split"),
    IntegerVar(1, 10, name="min_samples_leaf"),
]

gradientboosting_hyperparams = [
    IntegerVar(10, 200, name="n_estimators"),
    FloatVar(0.001, 0.3, name="learning_rate"),
    IntegerVar(1, 15, name="max_depth"),
    FloatVar(0.1, 1.0, name="subsample"),
    IntegerVar(2, 10, name="min_samples_split"),
    IntegerVar(1, 10, name="min_samples_leaf"),
    FloatVar(0.1, 1.0, name="max_features"),
]

histgradientboosting_hyperparams = [
    IntegerVar(10, 300, name="max_iter"),
    FloatVar(0.001, 0.3, name="learning_rate"),
    IntegerVar(10, 255, name="max_leaf_nodes"),
    FloatVar(0.1, 1.0, name="l2_regularization"),
    IntegerVar(1, 100, name="min_samples_leaf"),
]

algorithm_settings = {
    "CatBoostModel": {
        "bounds": catboost_hyperparams,
        "hyperparameters": {
            "depth": "depth",
            "learning_rate": "learning_rate",
            "iterations": "iterations",
            "l2_leaf_reg": "l2_leaf_reg",
            "border_count": "border_count",
            "bagging_temperature": "bagging_temperature",
            "random_strength": "random_strength",
        },
    },
    "AdaBoostModel": {
        "bounds": adaboost_hyperparams,
        "hyperparameters": {
            "n_estimators": "n_estimators",
            "learning_rate": "learning_rate",
        },
    },
    "LGBMModel": {
        "bounds": lgbm_hyperparams,
        "hyperparameters": {
            "num_leaves": "num_leaves",
            "max_depth": "max_depth",
            "learning_rate": "learning_rate",
            "n_estimators": "n_estimators",
            "subsample": "subsample",
            "colsample_bytree": "colsample_bytree",
            "reg_alpha": "reg_alpha",
            "reg_lambda": "reg_lambda",
        },
    },
    "XGBoostModel": {
        "bounds": xgboost_hyperparams,
        "hyperparameters": {
            "max_depth": "max_depth",
            "learning_rate": "learning_rate",
            "n_estimators": "n_estimators",
            "subsample": "subsample",
            "colsample_bytree": "colsample_bytree",
            "reg_alpha": "reg_alpha",
            "reg_lambda": "reg_lambda",
            "gamma": "gamma",
            "min_child_weight": "min_child_weight",
        },
    },
    "ExtraTreesModel": {
        "bounds": extratrees_hyperparams,
        "hyperparameters": {
            "n_estimators": "n_estimators",
            "max_depth": "max_depth",
            "max_features": "max_features",
            "min_samples_split": "min_samples_split",
            "min_samples_leaf": "min_samples_leaf",
        },
    },
    "GradientBoostingModel": {
        "bounds": gradientboosting_hyperparams,
        "hyperparameters": {
            "n_estimators": "n_estimators",
            "learning_rate": "learning_rate",
            "max_depth": "max_depth",
            "subsample": "subsample",
            "min_samples_split": "min_samples_split",
            "min_samples_leaf": "min_samples_leaf",
            "max_features": "max_features",
        },
    },
    "HistGradientBoostingModel": {
        "bounds": histgradientboosting_hyperparams,
        "hyperparameters": {
            "max_iter": "max_iter",
            "learning_rate": "learning_rate",
            "max_leaf_nodes": "max_leaf_nodes",
            "l2_regularization": "l2_regularization",
            "min_samples_leaf": "min_samples_leaf",
        },
    },
}
