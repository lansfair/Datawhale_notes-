from CART_clf import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.datasets import make_classification
import numpy as np

if __name__ == '__main__':

    X,y=make_classification(n_samples=200, n_features=8, n_informative=4, random_state=2021)
    # print(X[109], y[109])
    # print(X[1], y[1])
    my_cart=DecisionTreeClassifier(max_depth=3)
    my_cart.fit(X,y)
    res1=my_cart.predict(X)
    importance1 = my_cart.feature_importances_

    sklearn_cart=dt(max_depth=3)
    sklearn_cart.fit(X,y)
    res2=sklearn_cart.predict(X)
    importance2=sklearn_cart.feature_importances_

    print(res1)
    print(res2)
    idx = np.where(res1!=res2)
    print(idx)
    print(res1==res2)
    ###为什么有一个分类错误呢？
    print('结果一样的比例：',(np.abs(res1-res2) < 1e-8).mean())
    print('重要性一致的比例：',(np.abs(importance1-importance2)<1e-8).mean())