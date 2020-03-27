from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=0, test_size=0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
print(cm)
print(classification_report(Y_test, y_pred))
# 预测集中的0总共有68个，1总共有32个。 在这个混淆矩阵中，实际有68个0，
# 但K-NN预测出有67(64+3)个0，其中有3个实际上是1。 同时K-NN预测出有33(4+29)个1，其中4个实际上是0
