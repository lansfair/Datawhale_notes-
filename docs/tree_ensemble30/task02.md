# Task02 决策树（下）
## 1.CART代码实现
- 本次课程讲解了**分类回归决策树**（CART算法）的回归简单实现，加深了我对决策树理论和实现的理解。
-代码主要使用python中的numpy库实现。包括决策树类DecisionTreeRegressor，树类Tree，节点类Node，以及损失函数MSE

- 要实现决策树，首先需要定义一个决策树类DecisionTreeRegressor，其基于类Tree，简单类比skearn，需要实现**fit** （训练）和 **predict**（预测） 两个方法。

    - **fit**方法须传入feature：X和label：y，然后基于tree中的build方法构建决策树，再计算并保存每个feature的重要性。
重难点在于build方法的实现，在build方法中，
        - 1.首先定义build_prepare()方法，构建出初始状态，包括初始化depth为0，feature_importances_为0（长度为feature的数目），以及根节点root，root基于Node类，Node类属性包括当前depth，归属当前节点的样本编码idx（一个长为样本数量的样本，归属当前的样本对应位置为1，不属于则为0），以及左子节点left，右子节点right，当前分割特征feature，分割节点pivot
        - 2.定义build_node()方法，采用深度优先生长方法构建决策树叶子节点，直到达到最高深度或者当前节点样本数少于2或者某个子节点分配不到样本为止，到此训练完成。
            - 首先定义_able_to_split()方法判断当前节点是否符合分裂节点的条件
	        - 主要重难点在于如何分裂节点。定义split()方法分裂节点，基于均方误差计算H(Y)以及通过_get_conditional_entropy方法计算最小的H(Y|X)（返回H(Y|X)值，左子节点idx，右子节点idx，所选特征，分裂点）(_inner_split()方法使用最佳分割点，循环计算以每个样本点为分割点的H(Y|X)，返回当前循环特征的最小值)，从而计算出信息增益，再计算出相对信息增益并更新特征重要性，再新建左右节点并更新深度，返回值为（左子节点idx，右子节点idx，所选特征，分裂点）。

    - **predict** 方法实现较为简单，根据生成的决策树递归判断所预测的数据的所属节点，直至达到最下层节点为止，取当前节点所述样本的平均值作为回归预测值，预测完成。

- CART决策树回归应用具体实现代码如下：
```python
import numpy as np
```
```python
def MSE(y):
    return ((y - y.mean())**2).sum() / y.shape[0]
```

```python 
class Node:

    def __init__(self, depth, idx):
        self.depth = depth
        self.idx = idx

        self.left = None
        self.right = None
        self.feature = None
        self.pivot = None
```

```python
class Tree:

    def __init__(self, max_depth):
        self.max_depth = max_depth

        self.X = None
        self.y = None
        self.feature_importances_ = None

    def _able_to_split(self, node):
        return (node.depth < self.max_depth) & (node.idx.sum() >= 2)

    def _get_inner_split_score(self, to_left, to_right):
        total_num = to_left.sum() + to_right.sum()
        left_val = to_left.sum() / total_num * MSE(self.y[to_left])
        right_val = to_right.sum() / total_num * MSE(self.y[to_right])
        return left_val + right_val

    def _inner_split(self, col, idx):
        data = self.X[:, col]
        best_val = np.infty
        for pivot in data[:-1]:
            to_left = (idx==1) & (data<=pivot)
            to_right = (idx==1) & (~to_left)
            if to_left.sum() == 0 or to_left.sum() == idx.sum():
                continue
            Hyx = self._get_inner_split_score(to_left, to_right)
            if best_val > Hyx:
                best_val, best_pivot = Hyx, pivot
                best_to_left, best_to_right = to_left, to_right
        return best_val, best_to_left, best_to_right, best_pivot

    def _get_conditional_entropy(self, idx):
        best_val = np.infty
        for col in range(self.X.shape[1]):
            Hyx, _idx_left, _idx_right, pivot = self._inner_split(col, idx)
            if best_val > Hyx:
                best_val, idx_left, idx_right = Hyx, _idx_left, _idx_right
                best_feature, best_pivot = col, pivot
        return best_val, idx_left, idx_right, best_feature, best_pivot

    def split(self, node):
        # 首先判断本节点是不是符合分裂的条件
        if not self._able_to_split(node):
            return None, None, None, None
        # 计算H(Y)
        entropy = MSE(self.y[node.idx==1])
        # 计算最小的H(Y|X)
        (
            conditional_entropy,
            idx_left,
            idx_right,
            feature,
            pivot
        ) = self._get_conditional_entropy(node.idx)
        # 计算信息增益G(Y, X)
        info_gain = entropy - conditional_entropy
        # 计算相对信息增益
        relative_gain = node.idx.sum() / self.X.shape[0] * info_gain
        # 更新特征重要性
        self.feature_importances_[feature] += relative_gain
        # 新建左右节点并更新深度
        node.left = Node(node.depth+1, idx_left)
        node.right = Node(node.depth+1, idx_right)
        self.depth = max(node.depth+1, self.depth)
        return idx_left, idx_right, feature, pivot

    # 生成初始根节点
    def build_prepare(self):
        self.depth = 0
        self.feature_importances_ = np.zeros(self.X.shape[1])
        self.root = Node(depth=0, idx=np.ones(self.X.shape[0]) == 1)
    # 递归生成叶子结点
    def build_node(self, cur_node):
        if cur_node is None:
            return
        idx_left, idx_right, feature, pivot = self.split(cur_node)
        cur_node.feature, cur_node.pivot = feature, pivot
        self.build_node(cur_node.left)
        self.build_node(cur_node.right)

    def build(self):
        self.build_prepare()
        self.build_node(self.root)

    def _search_prediction(self, node, x):
        if node.left is None and node.right is None:
            return self.y[node.idx].mean()
        if x[node.feature] <= node.pivot:
            node = node.left
        else:
            node = node.right
        return self._search_prediction(node, x)

    def predict(self, x):
        return self._search_prediction(self.root, x)
```

```python
class DecisionTreeRegressor:
    """
    max_depth控制最大深度，类功能与sklearn默认参数下的功能实现一致
    """
    def __init__(self, max_depth):
        self.tree = Tree(max_depth=max_depth)

    def fit(self, X, y):
        self.tree.X = X
        self.tree.y = y
        self.tree.build()
        self.feature_importances_ = (
            self.tree.feature_importances_ 
            / self.tree.feature_importances_.sum()
        )
        return self

    def predict(self, X):
        return np.array([self.tree.predict(x) for x in X])
```
## 2.测试实现效果
- 测试是否与sklearn默认参数下实现的效果相同
- 注：测试时有时会产生两者结果预测部分不一致的情况，这种现象主要来自于当前节点在分裂的时候不同的特征和分割点组合产生了相同的信息增益，但由于遍历特征的顺序（和sklearn内的遍历顺序）不一样，因此在预测时会产生差异，并不是算法实现上有问题
```python
from CART import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor as dt
from sklearn.datasets import make_regression
```
```python
if __name__ == "__main__":

    # 模拟回归数据集
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, random_state=0
    )

    my_cart = DecisionTreeRegressor(max_depth=2)
    my_cart.fit(X, y)
    res1 = my_cart.predict(X)
    importance1 = my_cart.feature_importances_

    sklearn_cart = dt(max_depth=2)
    sklearn_cart.fit(X, y)
    res2 = sklearn_cart.predict(X)
    importance2 = sklearn_cart.feature_importances_

    # 预测一致的比例
    print(((res1-res2)<1e-8).mean())
    # 特征重要性一致的比例
    print(((importance1-importance2)<1e-8).mean())
```
## 3.修改为解决多分类问题
- 如果要实现分多分类问题，只需将计算信息增益的指标由均方误差更换为类别特征的基尼系数公式，并对最后预测值取arcmax即可。
```python
def Gini(y):
    gn=1.0
    n=y.shape[0]
    for i in np.unique(y):
        gn=gn-(np.sum(y==i)/n)**2
    return gn
```
- 对预测值取argmax，而不是像回归问题那样取平均值
```python
    def _search_prediction(self, node, x):
        if node.left is None and node.right is None:
            # return argmax(self.y[node.idx])
            return np.argmax(np.bincount(self.y[node.idx]))
            # return self.y[node.idx].min()
        if x[node.feature] <= node.pivot:
            node = node.left
        else:
            node = node.right
        return self._search_prediction(node, x)
```