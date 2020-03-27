import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,  4].values
print(X[:10])
print(Y)
# 将类别数据数字化
labelencode = LabelEncoder()
X[:, 3] = labelencode.fit_transform(X[:, 3])
print("labelencode\n", X[:10])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
print(X[:10])
# 此时要注意了：你用Onehotencode编码生成了3个特征变量，
# 这会造成虚拟变量陷阱，应该去掉一列，即构造变量个数=种类数-1
X = X[:, 1:]

# 拆分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=0, test_size=0.2)
# 训练数据
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
print(regressor.coef_, regressor.intercept_)
print(regressor.score(X_test, Y_test))
