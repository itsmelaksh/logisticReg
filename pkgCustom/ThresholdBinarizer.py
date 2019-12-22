from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, X, y):
        self.X_test = X
        self.y_test = y

    def Gini(self, y_pred):
        # check and get number of samples
        assert self.y_test.shape == y_pred.shape
        n_samples = self.y_test.shape[0]

        # sort rows on prediction column
        # (from largest to smallest)
        arr = np.array([self.y_test, y_pred]).transpose()
        true_order = arr[arr[:, 0].argsort()][::-1, 0]
        pred_order = arr[arr[:, 1].argsort()][::-1, 0]

        # get Lorenz curves
        L_true = np.cumsum(true_order) / np.sum(true_order)
        L_pred = np.cumsum(pred_order) / np.sum(pred_order)
        L_ones = np.linspace(1 / n_samples, 1, n_samples)

        # get Gini coefficients (area between curves)
        G_true = np.sum(L_ones - L_true)
        G_pred = np.sum(L_ones - L_pred)

        # normalize to true Gini coefficient
        return (G_pred / G_true)

    def accuracy(self, y_pred, probab_threshold=0.5):  # default threshold value is 0.5
        predicted_classes = (y_pred[:, 1] > probab_threshold).astype(int)
        return predicted_classes
