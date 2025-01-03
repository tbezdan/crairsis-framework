from mealpy import Problem
from utils.config import algorithm_settings
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from utils.logger import setup_logger
import numpy as np

logger = setup_logger(__name__)


class BaseOptimizerProblem(Problem):
    def __init__(
        self,
        bounds=None,
        minmax=None,
        X=None,
        y=None,
        ml_model_name=None,
        ml_model_constructor=None,
        **kwargs,
    ):
        self.X = X
        self.y = y
        self.ml_model_constructor = ml_model_constructor
        self.ml_model_name = ml_model_name
        self.task_type = kwargs.get("task_type", "regression")

        self.labels = [param.name for param in bounds]
        logger.info(f"Initializing BaseOptimizerProblem with cross-validation")

        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, solution):

        hyperparameter_values = self.decode_solution(solution)

        hyperparameter_config = algorithm_settings[self.ml_model_name][
            "hyperparameters"
        ]

        selected_hyperparameters = {
            name: hyperparameter_values[idx]
            for name, idx in hyperparameter_config.items()
        }

        model_instance = self.ml_model_constructor(**selected_hyperparameters)

        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            scoring_metric = "neg_log_loss"

            scores = cross_val_score(
                model_instance, self.X, self.y, cv=cv, scoring=scoring_metric
            )
            score = -np.mean(scores)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring_metric = "neg_mean_squared_error"

            scores = cross_val_score(
                model_instance, self.X, self.y, cv=cv, scoring=scoring_metric
            )

            score = -np.mean(scores)

        return score


class RegressionOptimizerProblem(BaseOptimizerProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type = "regression"


class ClassificationOptimizerProblem(BaseOptimizerProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type = "classification"
