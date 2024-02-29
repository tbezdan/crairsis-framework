import os
import sys
import contextlib
import lightgbm as lgb
from ml_models.base_model import BaseModel


class LGBMModel(BaseModel):
    def __init__(self, random_seed=42, **kwargs):
        super().__init__()
        # Set the random seed in kwargs if not already set
        kwargs.setdefault("random_state", random_seed)
        # Initialize LGBMRegressor with default parameters
        # Any parameters passed via kwargs will override LGBM's defaults
        self.model = lgb.LGBMRegressor(**kwargs)

    @contextlib.contextmanager
    def suppress_stdout_stderr(self):
        """
        A context manager that redirects stdout and stderr to devnull
        """
        with open(os.devnull, "w") as fnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = fnull, fnull
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    def train(self, X_train, y_train, **kwargs):
        """
        Train the LGBM model on the given dataset.
        """
        with self.suppress_stdout_stderr():
            self.model.fit(X_train, y_train, **kwargs)

    def evaluate(self, X_test, y_test, **kwargs):
        """
        Evaluate the LGBM model on the given test dataset.
        """
        with self.suppress_stdout_stderr():
            return self.model.score(X_test, y_test)

    def get_sklearn_estimator(self, random_seed=42, **kwargs):
        """
        Get a new instance of the sklearn estimator with specified hyperparameters.

        Parameters:
        - kwargs: Additional hyperparameters to pass to the model constructor.

        Returns:
        - A new instance of the sklearn estimator with specified hyperparameters.
        """
        kwargs.setdefault("random_state", random_seed)
        kwargs.setdefault("verbosity", -1)  # Keep the verbosity setting as well
        return lgb.LGBMRegressor(**kwargs)
