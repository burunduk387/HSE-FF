import forest 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

"""Вот тут просто Дз, всё самое вкусное в отдельных файлах, вызывается оно в конце файлика
и только если убрать #. Если удобно, можно также файлы sdss_nn_predict и sdss_full_predict 
попросить посчитать mse и написать кто из них троих оказался лучшим? :)"""  
#Pandas обусловлен удобством, на время выполнения он слабо влияет, главная беда 
#Очень медленное дерево (но оно с семнинара...)
data_train = pd.read_csv("sdss_redshift.csv")
X = data_train[["u", "g", "r", "i", "z"]].to_numpy()
y = data_train["redshift"].to_numpy()
data_test = pd.read_csv("sdss.csv")
#Ура ура в этом коде будет островок смысла
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
model = forest.RandomForest(max_depth=20, n_estimators=50)
model.fit(X_train, y_train)
txt = {"train": np.std(y_train - model.predict(X_train)),
       "test": np.std(y_test - model.predict(X_test))}
with open("redshift.json", "w") as f:
    json.dump(txt, f, separators=(",", ": "))
data_test["redshift"] = pd.Series(model.predict(data_test[["u", "g", "r", "i", "z"]].to_numpy()))
data_test.to_csv("sdss_predict.csv", index=False)
plt.figure()
plt.scatter(y_train, model.predict(X_train), color="palegoldenrod", label="Train")
plt.scatter(y_test, model.predict(X_test), color="green", alpha=0.3, label="Test")
plt.xlabel("True value")
plt.ylabel("Prediction")
plt.title("Forest prediction")
plt.legend()
plt.savefig("redshift.png")
#Казалось бы, почему бы после подбора гиперпараметров не обучиться на всех данных 
#и дальше жить именно так?
#Вот мне кажется так ОК делать (по крайней мере на ИАДе так делать точно разрешалось)
#Но М. Корнилов сказал не надо, но мне всё ещё интересно MSE такого подхода :)
def CallRFFull(X, y, X_train, y_train, X_test, y_test, data_test):
    model = forest.RandomForest(max_depth=15, n_estimators=50)
    model.fit(X, y)
    txt = {"train": np.std(y_train - model.predict(X_train)),
           "test": np.std(y_test - model.predict(X_test))}
    with open("redshift_full.json", "w") as f:
        json.dump(txt, f, separators=(",", ": "))
    data_test["redshift"] = pd.Series(model.predict(data_test[["u", "g", "r", "i", "z"]].to_numpy()))
    data_test.to_csv("sdss_full_predict.csv", index=False)
    plt.figure()
    plt.scatter(y_train, model.predict(X_train), color="palegoldenrod", label="Train")
    plt.scatter(y_test, model.predict(X_test), color="green", alpha=0.3, label="Test")
    plt.xlabel("True value")
    plt.ylabel("Prediction")
    plt.title("Forest prediction")
    plt.legend()
    plt.savefig("redshift_full.png")
def CallNeuro(X_train, y_train, X_test, y_test, data_test):
    import NN
    model = NN.NN_model()
    model.fit(X_train, X_test, y_train, y_test)
    txt = {"train": np.std(y_train - model.predict(X_train)),
       "test": np.std(y_test - model.predict(X_test))}
    with open("redshift_neuro.json", "w") as f:
        json.dump(txt, f, separators=(",", ": "))
    data_test["redshift"] = pd.Series(model.predict(data_test[["u", "g", "r", "i", "z"]].to_numpy()).reshape(data_test.shape[0]))
    data_test.to_csv("sdss_nn_predict.csv", index=False)
    plt.figure()
    plt.scatter(y_train, model.predict(X_train), color="blue", alpha=0.3, label="Train")
    plt.scatter(y_test, model.predict(X_test), color="red", alpha=0.3, label="Test")
    plt.xlabel("True value")
    plt.ylabel("Prediction")
    plt.title("NN model prediction")
    plt.legend()
    plt.savefig("redshift_neuro.png")
#Вот этих товарищей надо лишить символа комментария для веселья
#CallRFFull(X, y, X_train, y_train, X_test, y_test, data_test)  
#CallNeuro(X_train, y_train, X_test, y_test, data_test)
