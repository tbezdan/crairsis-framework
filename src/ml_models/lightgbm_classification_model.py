from lightgbm import LGBMClassifier
from ml_models.base_model import BaseModel
import os
import sys
import contextlib


class LGBMClassificationModel(BaseModel):
    def __init__(self, random_seed=42, **kwargs):
        super().__init__()
        kwargs.setdefault("class_weight", "balanced")
        kwargs.setdefault("random_state", random_seed)
        kwargs.setdefault("verbosity", -1)  # Suppress warnings by default
        self.model = LGBMClassifier(**kwargs)

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
        with self.suppress_stdout_stderr():
            self.model.fit(X_train, y_train, **kwargs)

    def evaluate(self, X_test, y_test, **kwargs):
        with self.suppress_stdout_stderr():
            return self.model.score(X_test, y_test)

    def predict_proba(self, X):
        with self.suppress_stdout_stderr():
            return self.model.predict_proba(X)

    def get_sklearn_estimator(self, random_seed=42, **kwargs):
        kwargs.setdefault("class_weight", "balanced")
        kwargs.setdefault("random_state", random_seed)
        kwargs.setdefault("verbosity", -1)  # Keep the verbosity setting as well
        return LGBMClassifier(**kwargs)
