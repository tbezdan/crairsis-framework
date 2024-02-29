from sklearn.ensemble import GradientBoostingRegressor
from ml_models.base_model import BaseModel


class GradientBoostingModel(BaseModel):
    def __init__(self, random_seed=42, **kwargs):
        super().__init__()
        # Set the random seed in kwargs if not already set
        kwargs.setdefault("random_state", random_seed)
        # Initialize GradientBoostingRegressor with default parameters
        # Any parameters passed via kwargs will override Gradient Boosting's defaults
        self.model = GradientBoostingRegressor(**kwargs)

    def train(self, X_train, y_train, **kwargs):
        """
        Train the Gradient Boosting model on the given dataset.

        Parameters:
        - X_train: Features of the training set.
        - y_train: Targets of the training set.
        - kwargs: Additional arguments to pass to the Gradient Boosting fit method.
        """
        self.model.fit(X_train, y_train, **kwargs)

    def evaluate(self, X_test, y_test, **kwargs):
        """
        Evaluate the Gradient Boosting model on the given test dataset.

        Parameters:
        - X_test: Features of the test set.
        - y_test: True targets of the test set.
        - kwargs: Additional arguments to pass to the evaluation method of Gradient Boosting.

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
        kwargs.setdefault("random_state", random_seed)
        return GradientBoostingRegressor(**kwargs)
