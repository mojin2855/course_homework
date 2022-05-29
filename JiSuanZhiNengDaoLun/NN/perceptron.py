# %%
import copy
from cmath import cos
import numpy as np
import pandas as pd
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#根据鸢尾花数据前三个特征，绘制三维分类散点图
x = np.array(df.iloc[:, [0]])
y = np.array(df.iloc[:, [1]])
z = np.array(df.iloc[:, [2]])
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
label_pred = []
for i in iris.target:
    if i == 0:
        label_pred.append('g')
    elif i == 1:
        label_pred.append('y')
    else:
        label_pred.append('b')

ax.scatter(x[:50], y[:50], z[:50], c=label_pred[:50], label='山鸢尾')
ax.scatter(x[50:100], y[50:100], z[50:100], c=label_pred[50:100], label='杂色鸢尾')
ax.scatter(x[100:], y[100:], z[100:], c=label_pred[100:], label='维吉尼亚鸢尾')
ax.set_xlabel('萼片长度 (cm)', fontdict={'size': 10, 'color': 'black'})
ax.set_ylabel('萼片宽度 (cm)', fontdict={'size': 10, 'color': 'black'})
ax.set_zlabel('花瓣长度 (cm)', fontdict={'size': 10, 'color': 'black'})
plt.legend()


class Model:
    def __init__(self, dim):
        #初始化权重
        self.d = dim
        self.w = np.ones(dim, dtype=np.float32)
        # self.w =np.array([-0.1 , 0.3, -0.4])
        self.b = -0.5
        self.l_rate = 1

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    def pocket(self, w, b, X_train, y_train):
        wrong_number = 0
        for d in range(len(X_train)):
            if y_train[d] * self.sign(X_train[d], w, b) <= 0:
                wrong_number += 1
        return wrong_number

    def fit(self, X_train, y_train):
        wrong_number_last = 1000
        times = 0
        while True:
            times += 1
            self.l_rate *= 0.5*cos(times*8*np.pi/4300)+0.5

            y_pred = self.sign(X_train, self.w, self.b)
            y_pred[y_pred > 0] = 1
            y_pred[y_pred <= 0] = -1
            num_fault = len(np.where(y_train != y_pred)[0])
            if num_fault == 0:
                acc = 1
                break
            r = np.random.choice(num_fault)
            i = np.where(y != y_pred)[0][r]
            Xd = X_train[i]
            yd = y_train[i]
            if yd * self.sign(Xd, self.w, self.b) <= 0:
                w = self.w + self.l_rate * np.dot(yd, Xd)
                b = self.b + self.l_rate * yd
                wrong_number_now = self.pocket(w, b, X_train, y_train)
                if wrong_number_now < wrong_number_last:
                    self.w = w
                    self.b = b
                    wrong_number_last = wrong_number_now
            if times > 4300:
                acc = 1-num_fault/len(y_train)
                break
        return times, acc

    def predict(self, x):
        if self.sign(x, self.w, self.b) > 0:
            print("1")
        else:
            print("-1")


# %%
CLASS = 'not linear'
if CLASS == 'linear':
    #线性可分数据========================
    X = np.array(df.iloc[:100, [0, 1, 2]])
    y = copy.deepcopy(iris.target[:100])
    label1 = '山鸢尾'
    label2 = '杂色鸢尾'
    label_pred = ['g' if i == 1 else 'y' for i in y]
elif CLASS == 'not linear':
    #线性不可分数据======================
    X = np.array(df.iloc[50:, [0, 1, 2]])
    y = copy.deepcopy(iris.target[50:])
    label1 = '杂色鸢尾'
    label2 = '维吉尼亚鸢尾'
    label_pred = ['y' if i == 1 else 'b' for i in y]
# ==================================
y[:50] = 1
y[50:] = -1
acc = 0
while acc < 0.89:
    perceptron = Model(len(X[0]))
    times, acc = perceptron.fit(X, y)
print('w:', perceptron.w, 'b:', perceptron.b, 'cul_times=', times, '准确率：', acc)


x0 = X[:, 0]
y0 = X[:, 1]
z0 = X[:, 2]
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(x0[:50], y0[:50], z0[:50], c=label_pred[:50], label=label1)
ax.scatter(x0[50:], y0[50:], z0[50:], c=label_pred[50:], label=label2)
ax.set_xlabel('萼片长度 (cm)', fontdict={'size': 10, 'color': 'black'})
ax.set_ylabel('萼片宽度 (cm)', fontdict={'size': 10, 'color': 'black'})
ax.set_zlabel('花瓣长度 (cm)', fontdict={'size': 10, 'color': 'black'})
plt.legend()
x1 = np.arange(4, 10, 1)
y1 = np.arange(1, 6, 1)
x1, y1 = np.meshgrid(x1, y1)
z1 = -(perceptron.w[0]*x1+perceptron.w[1]*y1+perceptron.b)/perceptron.w[2]
ax.plot_surface(x1, y1, z1, color='#23074d', alpha=0.5)

plt.show()
