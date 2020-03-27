# 使用sklearn进行数据预处理
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('../datasets/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print("初始状态下的X：\n", X)
print("初始状态下的Y：\n", Y)
# 处理缺失值
imputer = Imputer(missing_values='NaN', strategy="mean", axis=0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
print("处理缺失值后的X：\n", X)
# 将标签值转化成数字
X[:, 0] = LabelEncoder().fit_transform(X[:, 0])
X_OneHotEncoder = OneHotEncoder(categorical_features=[0])
X = X_OneHotEncoder.fit_transform(X).toarray()
Y = LabelEncoder().fit_transform(Y)
print("标签转换后的X：\n", X)
print("标签转换后的Y：\n", Y)
# 拆分测试集和数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
# 特征标准化
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)
print("X_train")
print(X_train)
print("X_test")
print(X_test)
