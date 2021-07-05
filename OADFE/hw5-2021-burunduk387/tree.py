import numpy as np
from scipy import optimize
import warnings 
#Ну как обычно ага
warnings.filterwarnings('ignore')


#Беру дерево с семинара и меняю под регрессию
#По пути ещё и убрал всякие имхо лишние вещи, потому что задание
#Напишите дерево, а детали не указаны
class RegressionTree(object):
  
    
    def __init__(self, max_depth=np.inf):
        self.max_depth = max_depth
        self.start = None
        self.left = None
        self.right = None
        self.depth = 0
        self.Leaf = None
        self.feature = None
        self.threshold = None
        
    def fit(self, X, y):
        self.start = RegressionTree()
        self.start.max_depth = self.max_depth
        self.start.build_tree(X, y)

    def build_tree(self, X, y):
        #Критерий останова любезно позаимствован с семинара
        if self.depth > self.max_depth or self.mse_sse(y) < 1e-6:
            self.Leaf = np.mean(y)
            return
        feature, threshold = self.find_best_split(X, y)
        self.left, self.right = RegressionTree(self.max_depth), RegressionTree(self.max_depth)
        self.feature = feature
        self.threshold = threshold
        self.left.depth, self.right.depth = self.depth + 1, self.depth + 1
        self.left.build_tree(X[X[:, feature] < threshold], y[X[:, feature] < threshold])
        self.right.build_tree(X[X[:, feature] >= threshold], y[X[:, feature] >= threshold])

    def find_best_split(self, X, y):
        best_feature, best_threshold, best_sse = 0, X[0, 0], np.inf
        for feature in range(X.shape[1]):
            if X.shape[0] > 2:
                feature_level = np.sort(X[:, feature])
                res = optimize.minimize_scalar(
                    self.SSE,
                    args=(feature, X, y),
                    bounds=(feature_level[1], feature_level[-1]),
                    method='Bounded',
                )
                assert res.success
                new_threshold = res.x
                new_sse = res.fun 
            else:
                new_threshold = np.max(X[:, feature])
                new_sse = self.SSE(new_threshold, feature, X, y)
            if new_sse < best_sse:
                best_feature, best_threshold, best_sse = feature, new_threshold, new_sse
        return best_feature, best_threshold

    def SSE(self, threshold, feature, X, y):
        l = y[X[:, feature] < threshold]
        r = y[X[:, feature] >= threshold]
        return (self.mse_sse(l) * l.size + self.mse_sse(r) * r.size) / (l.size + r.size)
    
    def mse_sse(self, y):
        return np.sum((y - np.mean(y))**2)

    def predict(self, X):
        return np.array([self.start.lookup(el) for el in X])
    
    def lookup(self, X):
        if self.Leaf != None:
            return self.Leaf
        elif X[self.feature] < self.threshold:
            return self.left.lookup(X)
        else:
            return self.right.lookup(X)
    
    