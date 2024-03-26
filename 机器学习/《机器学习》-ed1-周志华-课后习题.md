# 《机器学习》-ed1-周志华-课后习题

## ch3-线性模型

1、什么情况下 $f(x) = \omega ^ T x + b$ 不必考虑偏置项？

2、证明对于参数 $\omega$ ，对率回归的目标函数 $y = \frac{1}{1+e^{\left(-\omega^T+b\right)}}$ 是非凸的，但其对数似然函数$l\left(\beta\right) = sum_{i=1}^m \left(-y_i \beta^T x_i + \ln\left(1 + e ^ {\beta^T x_i}\right) \right)$ 是凸的

3、编程实现对率回归

7、码长为9，类别数为4，试给出海明距离意义下理论最优的ECOC二元码并证明

## ch4-决策树

2、使用最小训练误差作为决策树划分选择准则的缺陷

answer：使用最小训练误差可能导致过拟合问题，在训练过程中倾向于选择能够最好拟合训练集数据的划分从而忽略了泛化能力。且容易收到每个样本的影响。

3、编程实现基于信息熵的决策树算法

```python
import numpy as np

class DecisionTree():
    def __init__(self) -> None:
        self.tree = None

    def entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p = np.mean(y == cls)
            entropy -= p * np.log2(p)
        return entropy
    
    def information_gain(self, X, y, feature_idx, threshold):
        left_indices = X[:, feature_idx] <= threshold
        right_indices = ~left_indices
        n = len(y)
        left_entropy = self.entropy(y[left_indices])
        right_entropy = self.entropy(y[right_indices])
        left_weight = np.sum(left_indices) / n
        right_weight = np.sum(right_indices) / n
        return self.entropy(y) - (left_weight * left_entropy + right_weight * right_entropy)
    
    def find_best_split(self, X, y):
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        return best_feature_idx, best_threshold
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return {"class": y[0]}
        
        best_feature_idx, best_threshold = self.find_best_split(X, y)
        left_indices = X[:, best_feature_idx] <= best_threshold
        right_indices = ~left_indices
        left_subtree = self._grow_tree(X[left_indices], y[left_indices])
        right_subtree = self._grow_tree(X[right_indices], y[right_indices])
        return {'feature_idx': best_feature_idx, 'threshold': best_threshold, 'left': left_subtree, 'right': right_subtree}
    

    def _predict_one(self, x, tree):
        if 'class' in tree:
            return tree['class']
        feature_value = x[tree['feature_idx']]
        if feature_value <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])
        
    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
    

if __name__ == "__main__":
    # Sample data
    X = np.array([[2, 3], [1, 5], [3, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    # Initialize and train decision tree model
    model = DecisionTree()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    print("Predictions:", y_pred)
    
```

4、使用基尼指数实现决策树

## ch10-降维与聚类

1、k近邻算法实现

```python
import numpy as np
from collections import Counter
class KNN:
    def__init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.x_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = []
        for smaple in X:
            distance = [np.linalg.norm(sample - x) for x in self.x_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels =[self.y_train[i] for i n nearest_indices]
            most_common_label = Counter(nearest_labels).most_common(1)[0][0]
X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
y_train = np.array(['A', 'A', 'B', 'B'])
X_test = np.array([[4, 5], [1, 1,]])

knn = KNN(2)
knn.fit(X_train, y_train)
predictions knn.predict(X_test)
print("predictions:", predictions)
            
        
```

