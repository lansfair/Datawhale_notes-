from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

m1 = KNeighborsRegressor()
m2 = DecisionTreeRegressor()
m3 = LinearRegression()

models = [m1, m2, m3]

final_model = DecisionTreeRegressor()

m = len(models)

if __name__ == "__main__":
    boston = load_boston()
    X_data = boston['data']
    y_data = boston['target']

    X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=0.25)
    X_train, final_X, y_train, final_y= train_test_split(X, y, test_size=0.5)

    final_train = pd.DataFrame(np.zeros((X_test.shape[0], m)))
    final_test = pd.DataFrame(np.zeros((final_X.shape[0], m)))

    for model_id in range(m):
        model = models[model_id]
        model.fit(X_train, y_train)
        final_train.iloc[:, model_id] = model.predict(X_test)
        final_test.iloc[:, model_id] += model.predict(final_X)
        print("模型{0}的均方误差{1}".format(model_id,
                                     mean_squared_error(model.predict(final_X), final_y)))

    final_model.fit(final_train, y_test)
    res = final_model.predict(final_test)
    print("模型融合后的均方误差", mean_squared_error(res, final_y))