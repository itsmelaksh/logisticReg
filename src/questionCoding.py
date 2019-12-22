from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, auc, accuracy_score
#from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
#from sklearn.linear_model import LogisticRegression
import numpy as np
#from pkgCustom.custom_estimator import custom_estimator
#from pkgCustom.ThresholdBinarizer import ThresholdBinarizer

from pkgCustom import custom_estimator
from pkgCustom import ThresholdBinarizer

iris = datasets.load_iris()
X = iris.data
y = iris.target

# to make the problem as binary classification converted greater than 1 to 1
y=np.where(y>1, 1, y)

#splitting the dataset in 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# import warnings filter
# to reduce the warning received while initializing model
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

modelLogisticRegression = custom_estimator(X_train, y_train)

modelLogisticRegression.fit()

predictions = modelLogisticRegression.predict(X_test)
metricEvaluation = ThresholdBinarizer(X_test, y_test)

predictions =modelLogisticRegression.predict_prob(X_test)
threshold = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
finalValue = []
for indThsld in threshold:
    #y_pred = np.where(predictions[:,1]>indThsld,1,0)
    thsld_pred = metricEvaluation.accuracy(predictions,indThsld)
    print(thsld_pred)
    finalValue.append(metricEvaluation.Gini(thsld_pred))

print(finalValue)