from joblib import dump, load


class BaseModel:
    def __init__(self):
        """
        Initialize the base model. This is a placeholder for the actual model instance,
        which should be set in the subclass.
        """
        self.model = None

    def train(self, X_train, y_train, **kwargs):
        """
        Train the model on the given dataset. This method should be implemented by subclasses.

        Parameters:
        - X_train: Features of the training set.
        - y_train: Targets of the training set.
        - kwargs: Additional arguments to pass to the training method of the specific model.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def predict(self, X):
        """
        Predict the target for the given input data.

        Parameters:
        - X: Input features to predict.

        Returns:
        - Predictions made by the model.
        """
        if self.model is None:
            raise Exception("Model has not been trained yet")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, **kwargs):
        """
        Evaluate the model on the given test dataset. This method should be implemented by subclasses.

        Parameters:
        - X_test: Features of the test set.
        - y_test: True targets of the test set.
        - kwargs: Additional arguments to pass to the evaluation method of the specific model.

        Returns:
        - Evaluation metrics (like accuracy, MSE, etc.) depending on the model and task.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def save(self, file_path):
        """
        Save the trained model to a file using joblib.

        Parameters:
        - file_path: Path to the file where the model should be saved.
        """
        if self.model is None:
            raise Exception("No model to save")
        dump(self.model, file_path)

    def load(self, file_path):
        """
        Load a model from a file using joblib.

        Parameters:
        - file_path: Path to the file from which the model should be loaded.
        """
        self.model = load(file_path)

    def get_sklearn_estimator(self):

        return self.model
