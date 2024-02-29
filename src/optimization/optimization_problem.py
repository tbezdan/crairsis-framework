from mealpy import Problem
from utils.config import algorithm_settings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from utils.logger import setup_logger
import numpy as np
from sklearn.model_selection import KFold

logger = setup_logger(__name__)


class RegressionOptimizerProblem(Problem):
    def __init__(
        self,
        bounds=None,
        minmax="min",
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

        # Initialize labels with hyperparameter names from bounds
        self.labels = [param.name for param in bounds]
        logger.info(f"Initializing RegressionOptimizerProblem with cross-validation")

        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, solution):
        # Decode the solution to get hyperparameter values
        hyperparameter_values = self.decode_solution(solution)

        # Get the hyperparameter configuration for the current model
        hyperparameter_config = algorithm_settings[self.ml_model_name][
            "hyperparameters"
        ]

        # Map the decoded values to their corresponding hyperparameter names as per the config
        selected_hyperparameters = {
            name: hyperparameter_values[idx]
            for name, idx in hyperparameter_config.items()
        }

        # logger.info(f"Evaluating with hyperparameters: {selected_hyperparameters}")

        # Create the model instance with the selected hyperparameters
        model_instance = self.ml_model_constructor(**selected_hyperparameters)

        # Perform cross-validation and compute the average MSE
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            model_instance, self.X, self.y, cv=cv, scoring="neg_mean_squared_error"
        )

        # Convert scores to positive MSE and calculate the mean
        mse = -np.mean(scores)

        return mse
