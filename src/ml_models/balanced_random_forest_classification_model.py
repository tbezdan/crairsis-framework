from imblearn.ensemble import BalancedRandomForestClassifier
from ml_models.base_model import BaseModel


class BalancedRandomForestCls(BaseModel):
    def __init__(self, random_seed=42, **kwargs):
        super().__init__()
        kwargs.setdefault(
            "sampling_strategy", "all"
        )  # Set to "all" to align with the future behavior
        kwargs.setdefault("replacement", True)  # Set to True for replacement sampling
        kwargs.setdefault(
            "bootstrap", False
        )  # Set to False to align with the future behavior
        kwargs.setdefault("random_state", random_seed)
        self.model = BalancedRandomForestClassifier(**kwargs)

    def train(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train, **kwargs)

    def evaluate(self, X_test, y_test, **kwargs):
        return self.model.score(X_test, y_test)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_sklearn_estimator(self, random_seed=42, **kwargs):
        kwargs.setdefault("sampling_strategy", "all")
        kwargs.setdefault("replacement", True)
        kwargs.setdefault("bootstrap", False)
        kwargs.setdefault("random_state", random_seed)
        return BalancedRandomForestClassifier(**kwargs)
