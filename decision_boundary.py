from sklearn import datasets
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
%matplotlib inline

#アヤメのデータを読み込む
iris = datasets.load_iris()
input_data = iris.data

#訓練用データと品種をセットする
X = iris.data[:, [0, 2]] 
y = iris.target

#グラフ用の初期設定
h = .02  
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#グラフ描画用関数
def decision_boundary(clf, X, y, ax, title):
    clf.fit(X, y)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

    ax.set_title(title)
    ax.set_xlabel('sepal length')
    ax.set_ylabel('petal length')
    
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

#K近傍法の実施とグラフの描画
for ax, n_neighbors in zip(axes, [1,3,6,10]):
    title = "%s neighbor(s)"% (n_neighbors)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    decision_boundary(clf, X, y, ax, title)
 
