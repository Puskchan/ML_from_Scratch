from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from Logistic_regression import LogisticRegression
from KNN_Classifier import KNN
import Regularization_class
# Number of samples
num_samples = 1000

# Generate features (X) - two-dimensional random data
X = np.random.randn(num_samples, 2)

# Generate labels (y) - binary classification (0 or 1)
y = np.random.randint(0, 2, size=num_samples)

# logreg = LogisticRegression(0.01, 1000, Regularization_class.l1_reg(0.1))
# logreg.fit(X, y)
# log_pred = logreg.predict(X)
# log_score = accuracy_score(y, log_pred)
# print("The accuracy_score of the trained model (logistic): ", log_score)

knn_X, knn_y = make_classification(n_samples=1000, n_classes=2)
knn_y = y[:, np.newaxis]



knn_cla = KNN(5, "Minkowski")
knn_cla.train(knn_X, knn_y) 
y_pred = knn_cla.predict(knn_X)
acc = np.sum(knn_y==y_pred)/X.shape[0]
print("Accuracy of the prediction is (knn){}".format(acc))

#sklearn algo

print(100*'#')

# lr = linear_model.LogisticRegression()
# lr.fit(X,y)
# lp = lr.predict(X)
# ls = accuracy_score(y, lp)
# print("The accuracy_score of the trained model (logistic): ", ls)


knn_sklearn = KNeighborsClassifier(n_neighbors=5)
knn_sklearn.fit(knn_X, knn_y)
y_pred_sklearn = knn_sklearn.predict(knn_X)
acc = accuracy_score(knn_y, y_pred_sklearn)
print("Accuracy of the prediction is (knn sklearn){}".format(acc))
