 data 第一列为标记值
# data 后几列为特征向量
# initialTheta 为需要求得的theta
#!/sur/bin/python3
#-*-coding=utf-8-*-
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
%matplotlib inline
plt.rcParams["font.family"] = 'Arial Unicode MS'
###################
## data 第一列为真值，后面所有列为特征
## initialTheta 估算的权值初值
## featureNum 特征的个数
def RLS_Fun(data, initialTheta, featureNum):
    Theta = initialTheta
    P = 10 ** 6 * np.eye(featureNum)
    lamda = 1
    for i in range(len(data)):
        featureMatrix = data[i][1:]
        featureMatrix = featureMatrix.reshape(featureMatrix.shape[0], 1)
        y_real = data[i][0]
        K = np.dot(P, featureMatrix) / (lamda + np.dot(np.dot(featureMatrix.T, P), featureMatrix))
        Theta = Theta + np.dot(K, (y_real - np.dot(featureMatrix.T, Theta)))
        P = np.dot((np.eye(featureNum) - np.dot(K, featureMatrix.T)), P)
    return Theta
 
 
if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    dataInitial = sklearn.datasets.load_boston()
    x = np.array(dataInitial.data)
    y = np.array(dataInitial.target)
    y = y.reshape((y.shape[0], 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    data = np.concatenate((y_train, x_train), axis=1)
    featureNum = np.shape(x)[1]  # 有几个特征
    initialTheta = 0.5 * np.ones((featureNum, 1))
    Theta = RLS_Fun(data, initialTheta, featureNum)
    y_pred = np.dot(x_test, Theta)
    mse = mean_squared_error(y_test, y_pred)
    print('均方误差：', mse)
    t = np.arange(len(y_pred))
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', lw=2, label='真实值')
    plt.plot(t, y_pred, 'g-', lw=2, label='估计值')
    plt.legend(loc='best')
    plt.title('波士顿房价预测', fontsize=18)
    plt.xlabel('样本编号', fontsize=15)
    plt.ylabel('房屋价格', fontsize=15)
    plt.grid()
    plt.show()

