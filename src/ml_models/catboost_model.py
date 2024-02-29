from catboost import CatBoostRegressor
from ml_models.base_model import BaseModel


class CatBoostModel(BaseModel):
    def __init__(self, random_seed=42, **kwargs):
        super().__init__()
        # Set the random seed in kwargs if not already set
        kwargs.setdefault("random_seed", random_seed)
        # Initialize CatBoostRegressor with default parameters
        # Any parameters passed via kwargs will override CatBoost's defaults
        self.model = CatBoostRegressor(**kwargs)

    def train(self, X_train, y_train, verbose=False, **kwargs):
        """
        Train the CatBoost model on the given dataset.

        Parameters:
        - X_train: Features of the training set.
        - y_train: Targets of the training set.
        - kwargs: Additional arguments to pass to the CatBoost fit method.
        """
        self.model.fit(X_train, y_train, verbose=verbose, **kwargs)

    def evaluate(self, X_test, y_test, **kwargs):
        """
        Evaluate the CatBoost model on the given test dataset.

        Parameters:
        - X_test: Features of the test set.
        - y_test: True targets of the test set.
        - kwargs: Additional arguments to pass to the evaluation method of CatBoost.

        Returns:
        - The score of the model on the provided test data.
        """
        return self.model.score(X_test, y_test)

    def get_sklearn_estimator(self, random_seed=42, **kwargs):
        """
        Get a new instance of the sklearn estimator with specified hyperparameters.

        Parameters:
        - kwargs: Additional hyperparameters to pass to the model constructor.

        Returns:
        - A new instance of the sklearn estimator with specified hyperparameters.
        """
        kwargs.setdefault("random_seed", random_seed)
        kwargs.setdefault("verbose", 0)
        return CatBoostRegressor(**kwargs)
