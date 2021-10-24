import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree


class RandomForest:
    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for tree_id in range(self.n_estimators):
            indexes = np.random.randint(0, X.shape[0], X.shape[0])
            random_X = X[indexes]
            random_y = y[indexes]
            tree = Tree(max_depth=3)
            tree.fit(random_X, random_y)
            self.trees.append(tree)

    def predict(self, X):
        results = []

        for x in X:
            result = []
            for tree in self.trees:
                result.append(tree.predict(x.reshape(1, -1))[0])

            results.append(np.argmax(np.bincount(result)))  # 返回该样本的预测结果，采取方案：多数投票
        return np.array(results)