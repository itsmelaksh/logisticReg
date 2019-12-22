from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class custom_estimator(BaseEstimator, ClassifierMixin):
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y
        self.logmodel = LogisticRegression()

    def fit(self):
        return self.logmodel.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.logmodel.predict(X)

    def predict_prob(self, X):
        return self.logmodel.predict_proba(X)
