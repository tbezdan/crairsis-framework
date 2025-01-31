from sklearn.ensemble import ExtraTreesClassifier
from ml_models.base_model import BaseModel

"""

kwargs.setdefault("class_weight", "balanced")


Automatically adjusts class weights inversely proportional to class frequencies in the data.
The weight for each class i is computed as: weight_i = N / (n_i * number_of_classes)
where N is the total number of samples, and n_i is the number of samples in class i.
Example: For a dataset with 1000 samples (100 in Class 0, 900 in Class 1), 
the weights would be: weight_0 = 1000 / (100 * 2) = 5, weight_1 = 1000 / (900 * 2) ≈ 0.56.
This ensures that minority classes receive higher weights, preventing bias towards majority classes.
During training, these weights are used to penalize misclassifications.
"""


class ExtraTreesClassificationModel(BaseModel):
    def __init__(self, random_seed=42, **kwargs):
        super().__init__()
        kwargs.setdefault("class_weight", "balanced")
        kwargs.setdefault("random_state", random_seed)
        self.model = ExtraTreesClassifier(**kwargs)

    def train(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train, **kwargs)

    def evaluate(self, X_test, y_test, **kwargs):
        return self.model.score(X_test, y_test)

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    def get_sklearn_estimator(self, random_seed=42, **kwargs):
        kwargs.setdefault("class_weight", "balanced")
        kwargs.setdefault("random_state", random_seed)
        return ExtraTreesClassifier(**kwargs)
