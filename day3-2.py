from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

# 获取数据
boston = load_boston()

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(
    boston.data, boston.target, random_state=8)

# 特征工程，标准化
# 1> 创建一个转换器
transfer = StandardScaler()
# 2> 数据标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 方法一：正规方程求解
# 模型训练
# 1> 创建一个估计器
estimator_1 = LinearRegression()
# 2> 传入训练数据，进行机器学习
estimator_1.fit(x_train, y_train)
# 3> 打印梯度下降优化后的模型结果系数
print(estimator_1.coef_)
# 4> 打印梯度下降优化后的模型结果偏置
print(estimator_1.intercept_)

# 方法二：梯度下降求解
# 模型训练
# 1> 创建一个估计器，可以通过调参数，找到学习率效果更好的值
estimator_2 = SGDRegressor(learning_rate='constant', eta0=0.01)
# 2> 传入训练数据，进行机器学习
estimator_2.fit(x_train, y_train)
# 3> 打印梯度下降优化后的模型结果系数
print(estimator_2.coef_)
# 4> 打印梯度下降优化后的模型结果偏置
print(estimator_2.intercept_)

# 模型评估
# 使用均方误差对正规方程模型评估
y_predict = estimator_1.predict(x_test)
error = mean_squared_error(y_test, y_predict)
print('正规方程优化的均方误差为:\n', error)

# 使用均方误差对梯度下降模型评估
y_predict = estimator_2.predict(x_test)
error = mean_squared_error(y_test, y_predict)
print('梯度下降优化的均方误差为:\n', error)
