# Task03 集成模式
## 1. 为何集成
- 决策树的生成并不稳定，数据产生一定的噪音之后，整棵树构建出的样子可能会不一样。
- 可以通过训练多个决策树来提升稳定性：
    - 每棵树会独立的进行训练，训练之后这些树一起作用得出结果；
    - 分类的话，可以用投票（少数服从多数）；
    - 回归的话，可以对每棵树的结果求平均；
- 具体策略：
    - bagging：利用多个低偏差的学习器进行集成来降低模型的方差
    - boosting：或者利用多个低方差学习器进行集成来降低模型的偏差

## 2.bagging与boosting
- Bagging：在训练集中随机采样一些样本出来（放回，可重复）； 更进一步的随机森林算法，在bagging出来的数字中，再随机采样一些特征出来，不用整个特征。
- Boosting：
    - 顺序完成多个树的训练（串行集成，Bagging是独立的完成）
    - 利用训练好的树与真实值做残差来训练新的树，训练好了之后再与之前的树相加
    - 残差 等价于 取了一个平均均方误差（预测值与真实值的）再求梯度乘上个负号（因此叫做梯度提升树）


## 3.stacking与blending
- stacking本质上也属于并行集成方法，但其并不通过抽样来构造数据集进行基模型训练，而是采用K折交叉验证。
- blending在数据集上按照一定比例划分出训练集和验证集，每个基模型在训练集上进行训练。它的优势是模型的训练次数更少，但其缺陷在于不能使用全部的训练数据，相较于使用交叉验证的stacking稳健性较差。

## 4. 两种并行集成的树模型 
- 随机森林：以决策树（常用CART树）为基学习器的bagging算法
    - 回归问题：输出值为各学习器的均值
    - 分类问题：
        - 1.投票策略，即每个学习器输出一个类别，返回最高预测频率的类别（来自原始论文） 
        - 2.sklearn中采用的概率聚合策略，即通过各个学习器输出的概率分布先计算样本属于某个类别的平均概率，在对平均的概率分布取argmax以输出最可能的类别
    随机森林中的随机性来源：
        - bootstrap抽样导致的训练集随机性
        - 每个节点随机选取特征子集进行不纯度计算的随机性
        - 当使用随机分割点选取时产生的随机性（此时的随机森林又被称为Extremely Randomized Trees）
- 孤立森林：孤立森林也是一种使用树来进行集成的算法，其功能是用于连续特征数据的异常检测。
    - 孤立森林的基本思想是：多次随机选取特征和对应的分割点以分开空间中样本点，那么异常点很容易在较早的几次分割中就已经与其他样本隔开，正常点由于较为紧密故需要更多的分割次数才能将其分开。

## 5.侧边栏练习
$$
\begin{aligned}
L(\hat{f}) &= \mathbb{E}_D(y-\hat{f}_D)^2\\
&= \mathbb{E}_D(f+\epsilon-\hat{f}_D+\mathbb{E}_D[\hat{f}_{D}]-\mathbb{E}_D[\hat{f}_{D}])^2 \\
&= \mathbb{E}_D[(f-\mathbb{E}_D[\hat{f}_{D}])+(\mathbb{E}_D[\hat{f}_{D}]-\hat{f}_D)+\epsilon]^2 \\
&= \mathbb{E}_D[(f-\mathbb{E}_D[\hat{f}_{D}])^2] + \mathbb{E}_D[(\mathbb{E}_D[\hat{f}_{D}]-\hat{f}_D)^2] + \mathbb{E}_D[\epsilon^2] \\
&= [f-\mathbb{E}_D[\hat{f}_{D}]]^2 + \mathbb{E}_D[(\mathbb{E}_D[\hat{f}_{D}]-\hat{f}_D)^2] + \sigma^2
\end{aligned}
$$
1.上式第四个等号为何成立？
$$
\begin{aligned}
&=\mathbb{E}_{D}\left[\left(f-\mathbb{E}_{D}\left[\hat{f}_{D}\right]\right)+\left(\mathbb{E}_{D}\left[\hat{f}_{D}\right]-\hat{f}_{D}\right)+\epsilon\right]^{2} \\
&=\mathbb{E}_{D}\left[\left(f-\mathbb{E}_{D}\left[\hat{f}_{D}\right]\right)^{2}\right]+\mathbb{E}_{D}\left[\left(\mathbb{E}_{D}\left[\hat{f}_{D}\right]-\hat{f}_{D}\right)^{2}\right]+\mathbb{E}_{D}\left[\epsilon^{2}\right] \\
&+2 \mathbb{E}_{D}\left[\epsilon\left(f-\mathbb{E}_{D}\left[\hat{f}_{D}\right]\right)\right]+2 \mathbb{E}_{D}\left[\epsilon\left(\mathbb{E}_{D}\left[\hat{f}_{D}\right]-\hat{f}_{D}\right)\right] \\
&+2 \mathbb{E}_{D}\left[\left(f-\mathbb{E}_{D}\left[\hat{f}_{D}\right]\right)\left(\mathbb{E}_{D}\left[\hat{f}_{D}\right]-\hat{f}_{D}\right)\right]\\
\end{aligned}
$$ 

由于
$ \mathbb{E}_{D}[\epsilon]=0 $
且
$$ 
\begin{aligned}
& \mathbb{E}_{D}\left[\left(f-\mathbb{E}_{D}\left[\hat{f}_{D}\right]\right)\left(\mathbb{E}_{D}\left[\hat{f}_{D}\right]-\hat{f}_{D}\right)\right] \\
=& \mathbb{E}_{D}\left[f \mathbb{E}_{D}\left[\hat{f}_{D}\right]\right]-\mathbb{E}_{D}\left[\mathbb{E}_{D}\left[\hat{f}_{D}\right]^{2}\right]-\mathbb{E}_{D}\left[f \hat{f}_{D}\right]+\mathbb{E}_{D}\left[\mathbb{E}_{D}\left[\hat{f}_{D}\right] \hat{f}_{D}\right] \\
=& f \mathbb{E}_{D}\left[\hat{f}_{D}\right]-\mathbb{E}_{D}\left[\hat{f}_{D}\right]^{2}-f \mathbb{E}_{D}\left[\hat{f}_{D}\right]+\mathbb{E}_{D}\left[\hat{f}_{D}\right]^{2} \\
=& 0\\
\end{aligned}
$$ 
故上式
$$
\begin{aligned}
&=\mathbb{E}_{D}\left[\left(f-\mathbb{E}_{D}\left[\hat{f}_{D}\right]\right)^{2}\right]+\mathbb{E}_{D}\left[\left(\mathbb{E}_{D}\left[\hat{f}_{D}\right]-\hat{f}_{D}\right)^{2}\right]+\mathbb{E}_{D}\left[\epsilon^{2}\right] \\
\end{aligned}
$$
2.有人说[Bias-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)就是指“一个模型要么具有大的偏差，要么具有大的方差”，你认为这种说法对吗？你能否对“偏差-方差权衡”现象做出更准确的表述？

答：不完全正确，如果模型可以拟合绝大多数数据，那么偏差和方差都会在一个比较小的范围。

但由于模型总是无法完美拟合所有数据，在某种程度上二者存在不可调和的矛盾。

具体来说，

**偏差**度量了学习算法的期望预测与真实结果的偏离程度，刻画了学习算法本身的拟合能力。

**方差**度量了同样大小的训练集的变动所导致的学习性能的变化，刻画了数据扰动所造成的影响。

**噪声**表达了当前任务上任何学习算法所能达到的期望泛化误差的下界，也就是最小值。

泛化误差可以分解为偏差、方差和噪声之和。在训练达到一定的程度后，需要在偏差和方差之间有所权衡，使得泛化误差最低，即模型即有较高的预测精度，也有较强的泛化能力。

3.假设总体有$100$个样本，每轮利用bootstrap抽样从总体中得到$10$个样本（即可能重复），请求出所有样本都被至少抽出过一次的期望轮数。（通过[本文](https://en.wikipedia.org/wiki/Coupon_collector%27s_problem)介绍的方法，我们还能得到轮数方差的bound）

4.对于stacking和blending集成而言，若$m$个基模型使用$k$折交叉验证，此时分别需要进行几次训练和几次预测？

答：对于stacking 需要训练mk+1次，预测 2mk+1次。对于blending需要训练m+1次，预测2m+1次。

5.r2_score和均方误差的区别是什么？它具有什么优势？

6.假设使用闵氏距离来度量两个嵌入向量之间的距离，此时对叶子节点的编号顺序会对距离的度量结果有影响吗？

## 6. 知识回顾

1. 什么是偏差和方差分解？偏差是谁的偏差？此处的方差又是指什么？

    答：泛化误差可以分解为偏差、方差和噪声之和。偏差度量了学习算法的期望预测与真实结果的偏离程度，刻画了学习算法本身的拟合能力。方差度量了同样大小的训练集的变动所导致的学习性能的变化，刻画了数据扰动所造成的影响。

2. 相较于使用单个模型，bagging和boosting方法有何优势？

    答：bagging可以降低整体模型的方差，boosting可以降低整体模型的偏差。

3.请叙述stacking的集成流程，并指出blending方法和它的区别、

    后补

## 7. stacking
```python
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
 
import numpy as np
import pandas as pd
 
m1=KNeighborsRegressor()
m2=DecisionTreeRegressor()
m3=LinearRegression()
 
models=[m1,m2,m3]
 
#from sklearn.svm import LinearSVR
 
final_model=DecisionTreeRegressor()
 
k,m=4,len(models)
 
if  __name__=="__main__":
    X,y=make_regression(
        n_samples=1000,n_features=8,n_informative=4,random_state=0
    )
    final_X,final_y=make_regression(
        n_samples=500,n_features=8,n_informative=4,random_state=0
    )
    
    final_train=pd.DataFrame(np.zeros((X.shape[0],m)))
    final_test=pd.DataFrame(np.zeros((final_X.shape[0],m)))
    
    kf=KFold(n_splits=k)
    for model_id in range(m):
        model=models[model_id]
        for train_index,test_index in kf.split(X):
            X_train,X_test=X[train_index],X[test_index]
            y_train,y_test=y[train_index],y[test_index]
            model.fit(X_train,y_train)
            final_train.iloc[test_index,model_id]=model.predict(X_test)
            final_test.iloc[:,model_id]+=model.predict(final_X)
        final_test.iloc[:,model_id]/=k
    final_model.fit(final_train,y)
    res=final_model.predict(final_test)
```

## 8. blending
```python

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
 
import numpy as np
import pandas as pd
 
m1=KNeighborsRegressor()
m2=DecisionTreeRegressor()
m3=LinearRegression()
 
models=[m1,m2,m3]
 
final_model=DecisionTreeRegressor()
 
m=len(models)
 
if  __name__=="__main__":
    X,y=make_regression(
        n_samples=1000,n_features=8,n_informative=4,random_state=0
    )
    final_X,final_y=make_regression(
        n_samples=500,n_features=8,n_informative=4,random_state=0
    )
    
 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)
    
    final_train=pd.DataFrame(np.zeros((X_test.shape[0],m)))
    final_test=pd.DataFrame(np.zeros((final_X.shape[0],m)))
    
    for model_id in range(m):
        model=models[model_id]
        model.fit(X_train,y_train)
        final_train.iloc[:,model_id]=model.predict(X_test)
        final_test.iloc[:,model_id]+=model.predict(final_X)
        
    final_model.fit(final_train,y_train)
    res=final_model.predict(final_test)
```