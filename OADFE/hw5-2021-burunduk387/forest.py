import tree
import numpy as np
from threading import Thread


#А давайте сделаем многопоточность. Ну чтобы быстрее, плюс так дз интереснее
class ThreadTree(Thread):
    def __init__(self, X, y,  max_depth, trees):
        Thread.__init__(self)
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.trees = trees #Их надо храниить как-то, ничего умнее не придумал

    def run(self):
        tr = tree.RegressionTree(max_depth=self.max_depth)
        tr.fit(self.X, self.y)
        self.trees.append(tr)
        

class RandomForest(object):
    def __init__(self, max_depth=5, n_estimators=50):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.trees = []
       
        
    def fit(self, X, y):
        threads = []
        X = np.array(X)
        y = np.array(y)
        index = np.arange(0, X.shape[0])
        #Сказали можно бэггинг, значит напишем бэггинг
        for i in range(self.n_estimators):
            index_i = np.random.choice(index, X.shape[0])
            X_i, y_i = X[index_i], y[index_i]
            thread = ThreadTree(X_i, y_i, self.max_depth, self.trees)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()

    def predict(self, X_pred):
        predictions = np.zeros(X_pred.shape[0])
        for tr in self.trees:
            predictions = predictions + tr.predict(X_pred)
        return predictions / self.n_estimators
            
    
    



