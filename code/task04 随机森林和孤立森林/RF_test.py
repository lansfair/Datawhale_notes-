from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier as RF
from RandomForest_Classifier import RandomForest
import numpy as np

if __name__ == "__main__":
    X, y = make_classification(n_samples=200, n_features=8, n_informative=4, random_state=0)

    RF1 = RandomForest(n_estimators=100, max_depth=3)
    RF2 = RF(n_estimators=100, max_depth=3)

    RF1.fit(X, y)
    res1 = RF1.predict(X)

    RF2.fit(X, y)
    res2 = RF2.predict(X)

    print('结果一样的比例', (np.abs(res1 - res2) < 1e-5).mean())