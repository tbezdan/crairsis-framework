from sklearn.ensemble import HistGradientBoostingClassifier
from ml_models.base_model import BaseModel


class HistGradientBoostingCls(BaseModel):
    def __init__(self, random_seed=42, **kwargs):
        super().__init__()
        kwargs.setdefault("class_weight", "balanced")
        kwargs.setdefault("random_state", random_seed)
        self.model = HistGradientBoostingClassifier(**kwargs)

    def train(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train, **kwargs)

    def evaluate(self, X_test, y_test, **kwargs):
        return self.model.score(X_test, y_test)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_sklearn_estimator(self, random_seed=42, **kwargs):
        kwargs.setdefault("class_weight", "balanced")
        kwargs.setdefault("random_state", random_seed)
        return HistGradientBoostingClassifier(**kwargs)
