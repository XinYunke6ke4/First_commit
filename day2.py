from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv("../datasets/studentscores.csv")
print(dataset)

# X = dataset.iolc[:, 0].values,这种得到的是列表（一维),下面得到的是二维数据结构
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values
print("X:\n", X)
print("Y:\n", Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/4, random_state=0)
print(X_train)
print(Y_train)
# 训练线性回归
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
# 预测结果
Y_pred = regressor.predict(X_test)
print(regressor.coef_)
print(regressor.intercept_)
print(regressor.score(X_test, Y_test))
# 训练集结果和测试集结果可视化
plt.scatter(X_train, Y_train, color='Red')
plt.plot(X_train, regressor.predict(X_train), 'bo-')
plt.show()
plt.scatter(X_test, Y_test, color='Red')
plt.plot(X_test, Y_pred, 'bo-')
plt.show()
