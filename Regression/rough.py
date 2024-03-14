from sklearn.metrics import r2_score
from Regression import Linear_regression, Lasso_regression, Ridge_regression, Elasticnet_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=50000, n_features=8, noise=20)

print("="*100)
print('Our algos:')

lr = Linear_regression.LinearRegression(alpha=0.1, iter=100)
lr.fit(X,y)
y_pred = lr.predict(X)
score = r2_score(y, y_pred)
print("The r2_score of the trained model (linear): ", score)
# lr.show_line(X[:,1],y,lr.y_pred)

lasr = Lasso_regression.LassoRegression(0.1, 0.1, 100)
lasr.train(X,y)
y_pred_1 = lasr.predict(X)
score_lasso = r2_score(y, y_pred_1)
print("The r2_score of the trained model (lasso): ", score_lasso)
# lasr.show_line(X[:,1],y,lr.y_pred)

Rr = Ridge_regression.RidgeRegression(0.1, 0.1, 100)
Rr.train(X,y)
y_pred_2 = Rr.predict(X)
score_ridge = r2_score(y, y_pred_2)
print("The r2_score of the trained model (ridge): ", score_ridge)
# Rr.show_line(X[:,1],y,lr.y_pred)

Er = Elasticnet_regression.ElasticNet_reg(0.5, 0.1, 0.1, 100)
Er.train(X,y)
y_pred_3 = Er.predict(X)
score_elastic = r2_score(y, y_pred_3)
print("The r2_score of the trained model (elastic-net): ", score_elastic)

#######################################################

# Sklearn Models

#######################################################

print("="*100)
print('Sklearn algo:')

slr = LinearRegression()
slr.fit(X,y)
y_pred_sklearn = slr.predict(X)
score = r2_score(y, y_pred_sklearn)
print("R2 score of the model is (linear) {}".format(score))

llr = Lasso()
llr.fit(X,y)
y_pred_sklearn = llr.predict(X)
score = r2_score(y, y_pred_sklearn)
print("R2 score of the model is (lasso) {}".format(score))

rlr = Ridge()
rlr.fit(X,y)
y_pred_sklearn = rlr.predict(X)
score = r2_score(y, y_pred_sklearn)
print("R2 score of the model is (ridge) {}".format(score))

elr = ElasticNet()
elr.fit(X,y)
y_pred_sklearn = elr.predict(X)
score = r2_score(y, y_pred_sklearn)
print("R2 score of the model is (elastic-net) {}".format(score))

print("="*100)

