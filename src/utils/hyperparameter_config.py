from mealpy import FloatVar, IntegerVar


# Hyperparameters for Regression Models
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

# Hyperparameters for Classification Models
balancedrandomforest_hyperparams = [
    IntegerVar(50, 1000, name="n_estimators"),
    IntegerVar(1, 100, name="max_depth"),
    FloatVar(0.1, 1.0, name="max_features"),
    FloatVar(0.1, 1.0, name="min_samples_split"),
    FloatVar(0.1, 1.0, name="min_samples_leaf"),
    IntegerVar(2, 10, name="min_samples_split"),
    IntegerVar(1, 10, name="min_samples_leaf"),
]


catboost_classification_hyperparams = [
    IntegerVar(3, 10, name="depth"),
    FloatVar(0.001, 0.3, name="learning_rate"),
    IntegerVar(100, 1000, name="iterations"),
    FloatVar(1, 10, name="l2_leaf_reg"),
    FloatVar(0.01, 1, name="bagging_temperature"),
    FloatVar(1e-9, 10, name="random_strength"),
]

adaboost_classification_hyperparams = [
    IntegerVar(50, 500, name="n_estimators"),
    FloatVar(0.01, 1.0, name="learning_rate"),
]

lgbm_classification_hyperparams = [
    IntegerVar(10, 300, name="num_leaves"),
    IntegerVar(10, 100, name="max_depth"),
    FloatVar(0.001, 0.2, name="learning_rate"),
    IntegerVar(100, 2000, name="n_estimators"),
    FloatVar(0.6, 1.0, name="subsample"),
    FloatVar(0.6, 1.0, name="colsample_bytree"),
    FloatVar(0, 1, name="reg_alpha"),
    FloatVar(0, 1, name="reg_lambda"),
]

xgboost_classification_hyperparams = [
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

extratrees_classification_hyperparams = [
    IntegerVar(50, 1000, name="n_estimators"),
    IntegerVar(1, 100, name="max_depth"),
    FloatVar(0.1, 1.0, name="max_features"),
    IntegerVar(2, 10, name="min_samples_split"),
    IntegerVar(1, 10, name="min_samples_leaf"),
]

gradientboosting_classification_hyperparams = [
    IntegerVar(50, 200, name="n_estimators"),
    FloatVar(0.001, 0.3, name="learning_rate"),
    IntegerVar(1, 15, name="max_depth"),
    FloatVar(0.1, 1.0, name="subsample"),
    IntegerVar(2, 10, name="min_samples_split"),
    IntegerVar(1, 10, name="min_samples_leaf"),
    FloatVar(0.1, 1.0, name="max_features"),
]

histgradientboosting_classification_hyperparams = [
    IntegerVar(50, 300, name="max_iter"),
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
    "CatBoostClassificationModel": {
        "bounds": catboost_classification_hyperparams,
        "hyperparameters": {
            "depth": "depth",
            "learning_rate": "learning_rate",
            "iterations": "iterations",
            "l2_leaf_reg": "l2_leaf_reg",
            "bagging_temperature": "bagging_temperature",
            "random_strength": "random_strength",
        },
    },
    "AdaBoostClassificationModel": {
        "bounds": adaboost_classification_hyperparams,
        "hyperparameters": {
            "n_estimators": "n_estimators",
            "learning_rate": "learning_rate",
        },
    },
    "LGBMClassificationModel": {
        "bounds": lgbm_classification_hyperparams,
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
    "XGBClassificationModel": {
        "bounds": xgboost_classification_hyperparams,
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
    "ExtraTreesClassificationModel": {
        "bounds": extratrees_classification_hyperparams,
        "hyperparameters": {
            "n_estimators": "n_estimators",
            "max_depth": "max_depth",
            "max_features": "max_features",
            "min_samples_split": "min_samples_split",
            "min_samples_leaf": "min_samples_leaf",
        },
    },
    "GradientBoostingClassificationModel": {
        "bounds": gradientboosting_classification_hyperparams,
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
    "HistGradientBoostingCls": {
        "bounds": histgradientboosting_classification_hyperparams,
        "hyperparameters": {
            "max_iter": "max_iter",
            "learning_rate": "learning_rate",
            "max_leaf_nodes": "max_leaf_nodes",
            "l2_regularization": "l2_regularization",
            "min_samples_leaf": "min_samples_leaf",
        },
    },
    "BalancedRandomForestCls": {
        "bounds": balancedrandomforest_hyperparams,
        "hyperparameters": {
            "n_estimators": "n_estimators",
            "max_depth": "max_depth",
            "max_features": "max_features",
            "min_samples_split": "min_samples_split",
            "min_samples_leaf": "min_samples_leaf",
        },
    },
}
