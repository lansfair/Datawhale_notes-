from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

m1 = KNeighborsRegressor()
m2 = SVR()
m3 = LinearRegression()
# m3 = DecisionTreeRegressor()


models = [m1, m2, m3]

final_model = DecisionTreeRegressor()

k, m = 5, len(models)


if __name__ == "__main__":
    boston = load_boston()
    X_data = boston['data']
    y_data = boston['target']
    # SS = StandardScaler()
    # X_data = SS.fit_transform(X_data)
    X, final_X, y, final_y = train_test_split(X_data, y_data, test_size=0.2, random_state=2)
    final_train = pd.DataFrame(np.zeros((X.shape[0], m)))
    final_test = pd.DataFrame(np.zeros((final_X.shape[0], m)))

    kf = KFold(n_splits=k)
    for model_id in range(m):
        for train_index, test_index in kf.split(X):
            model = models[model_id]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            final_train.iloc[test_index, model_id] = model.predict(X_test)
            final_test.iloc[:, model_id] += model.predict(final_X)
        final_test.iloc[:, model_id] /= k
        print("模型{0}的均方误差{1}".format(model_id, mean_squared_error(final_test.iloc[:, model_id], final_y)))
    final_model.fit(final_train, y)
    res = final_model.predict(final_test)
    print("模型融合后的均方误差", mean_squared_error(res, final_y))