from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error


def linear_model1():
    """
    线性回归:正规方程
    """
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(
        boston.data, boston.target, test_size=0.2)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print("预测值为: \n", y_predict)
    print("模型中的系数为：\n", estimator.coef_)
    print("模型中的偏置为：\n", estimator.intercept_)
    error = mean_squared_error(y_test, y_predict)
    print("误差为：\n", error)
    return None


def linear_model2():
    """
    线性回归:梯度下降法
    """
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(
        boston.data, boston.target, test_size=0.2)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    estimator = SGDRegressor(max_iter=1000)
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print("预测值为: \n", y_predict)
    print("模型中的系数为：\n", estimator.coef_)
    print("模型中的偏置为：\n", estimator.intercept_)
    error = mean_squared_error(y_test, y_predict)
    print("误差为：\n", error)
    return None


def linear_model3():
    """
    线性回归:岭回归
    """
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(
        boston.data, boston.target, test_size=0.2)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    #estimator = Ridge(alpha=1)
    estimator = RidgeCV(alphas=(0.01, 0.1, 1, 10, 100))
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print("预测值为: \n", y_predict)
    print("模型中的系数为：\n", estimator.coef_)
    print("模型中的偏置为：\n", estimator.intercept_)
    error = mean_squared_error(y_test, y_predict)
    print("误差为：\n", error)
    return None


# linear_model1()
# linear_model2()
linear_model3()
