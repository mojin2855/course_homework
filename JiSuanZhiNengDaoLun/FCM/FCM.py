# %%
from sklearn import preprocessing, metrics
import numpy as np
from collections import Counter
from scipy.spatial import distance
import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white', {'font.sans-serif': ['simhei', 'Arial']})  # 解决中文不能显示问题

#数据可视化===================================================================================
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target_names[iris.target]
iris_data.head(3).append(iris_data.tail(3))

iris_data.rename(columns={
    "sepal length (cm)": "萼片长",
    "sepal width (cm)": "萼片宽",
    "petal length (cm)": "花瓣长",
    "petal width (cm)": "花瓣宽",
    "species": "种类"},
    inplace=True
)
iris_data.head(3).append(iris_data.tail(3))
kind_dict = {
    "setosa": "山鸢尾",
    "versicolor": "杂色鸢尾",
    "virginica": "维吉尼亚鸢尾"
}
iris_data["种类"] = iris_data["种类"].map(kind_dict)

sns.pairplot(iris_data, hue='种类')



# %%
#最大最小归一化================================================================
file = 'irisdata.csv'
data = np.genfromtxt(file, delimiter=',', dtype=None,
                     skip_header=1, usecols=(0, 1, 2, 3, 4))

# 对数据进行归一化处理
d_minMax = preprocessing.MinMaxScaler().fit_transform(data[:, 0:4])
d_01 = preprocessing.StandardScaler().fit_transform(data[:, 0:4])
d_test = d_minMax
# d_test = d_01

# 随机选取初始聚类中心
def init_center(data, k):
    len = data.shape[0]
    index = np.random.choice(np.arange(len), k)
    return data[index]

#初始化隶属度矩阵
def init_U(data_number, k):
    size = (data_number, k)
    U = np.random.uniform(0, 1, size)  # 模糊度均匀初始化
    U /= np.sum(U, axis=1, keepdims=True)
    return U

#更新聚类中心
def clustercenter(U, alpha, data, k):
    clu_center = [0]*k
    for i in range(k):
        uxsum = np.array([0, 0, 0, 0])
        for j in range(data.shape[0]):
            uxsum = (U[j, i]**alpha)*data[j, :]+uxsum
        clu_center[i] = np.array(uxsum)/sum(U[:, i]**alpha)
    return np.array(clu_center)

#更新隶属度矩阵
def lishu(data, clu_center, alpha, k):
    distk = [0]*k
    U = [0]*k
    clu_center = np.array(clu_center)
    for i in range(k):
        distk[i] = np.linalg.norm(data-clu_center[i],  axis=1, keepdims=True)
    distk = np.array(distk)
    distk = distk.reshape(3, 150)
    distk = distk.T+1e-10
    for i in range(k):
        U[i] = np.array((1 / (distk[:, i] ** (2/(alpha-1)))) /
                        (np.sum(1/(distk**(2/(alpha-1))), axis=1)))
    U = np.array(U)
    U[np.isnan(U)] = 1
    U = U.T
    return U


k = 3  # k 为要聚类的中心数
purity_all=[]
for alpha in range(2,17):
    rs_aver = 0
    purity_aver=0
    for n in range(10):
        clu_center = init_center(d_test, k)  # 聚类中心
        U = init_U(d_test.shape[0], k)  # 隶属度矩阵
        accuracy = 1
        dij = distance.cdist(d_test, clu_center, 'euclidean')
        times = 0
        while accuracy > 0.0001:
            times += 1
            if times > 1000:
                break
            clu_center_new = clustercenter(U, alpha, d_test, k)
            U = lishu(d_test, clu_center, alpha, k)
            accuracy = np.linalg.norm(
                clu_center_new[clu_center_new[:, 0].argsort()]-clu_center[clu_center[:, 0].argsort()])
            clu_center = clu_center_new

        clu_center = np.array(clu_center)
        kclass = np.argmax(U, axis=1)
        rs = metrics.rand_score(data[:, 4], kclass)
        A = data[kclass == 0]
        B = data[kclass == 1]
        C = data[kclass == 2]
        A_target = Counter(A[:, 4]).most_common(1)[0][0]
        # print(A_target)
        B_target = Counter(B[:, 4]).most_common(1)[0][0]
        # print(B_target)
        C_target = Counter(C[:, 4]).most_common(1)[0][0]
        # print(C_target)
        acc = Counter(A[:, 4]).most_common(1)[0][1] + \
            Counter(B[:, 4]).most_common(1)[0][1] + \
            Counter(C[:, 4]).most_common(1)[0][1]
        purity_aver+=acc
        rs_aver += rs
    purity_all.append(purity_aver/1500)
    print('alpha:',alpha, '10次FCM平均兰德指数为:', rs_aver/(n+1),'纯度为：',purity_aver/1500)


# %%01标准化=======================================================================
d_test = d_01

k = 3  # k 为要聚类的中心数
purity_all_01=[]
for alpha in range(2,17):
    rs_aver = 0
    purity_aver=0
    for n in range(10):
        clu_center = init_center(d_test, k)  # 聚类中心
        U = init_U(d_test.shape[0], k)  # 隶属度矩阵
        accuracy = 1
        dij = distance.cdist(d_test, clu_center, 'euclidean')
        times = 0
        while accuracy > 0.0001:
            times += 1
            if times > 1000:
                break
            clu_center_new = clustercenter(U, alpha, d_test, k)
            U = lishu(d_test, clu_center, alpha, k)
            accuracy = np.linalg.norm(
                clu_center_new[clu_center_new[:, 0].argsort()]-clu_center[clu_center[:, 0].argsort()])
            clu_center = clu_center_new

        clu_center = np.array(clu_center)
        kclass = np.argmax(U, axis=1)
        rs = metrics.rand_score(data[:, 4], kclass)
        A = data[kclass == 0]
        B = data[kclass == 1]
        C = data[kclass == 2]
        A_target = Counter(A[:, 4]).most_common(1)[0][0]
        # print(A_target)
        B_target = Counter(B[:, 4]).most_common(1)[0][0]
        # print(B_target)
        C_target = Counter(C[:, 4]).most_common(1)[0][0]
        # print(C_target)
        acc = Counter(A[:, 4]).most_common(1)[0][1] + \
            Counter(B[:, 4]).most_common(1)[0][1] + \
            Counter(C[:, 4]).most_common(1)[0][1]
        purity_aver+=acc
        rs_aver += rs
    purity_all_01.append(purity_aver/1500)
    print('alpha:',alpha, '10次FCM平均兰德指数为:', rs_aver/(n+1),'纯度为：',purity_aver/1500)

# %%参数选择===========================================================================
plt.figure()
plt.plot(range(2, 17), purity_all,label='最大最小归一化')
plt.plot(range(2, 17), purity_all_01, label='01标准化')
plt.xlabel('alpha')
plt.ylabel('纯度purity')
plt.legend()
plt.savefig('fcmalpha.svg')




# %%
#将参数调为最佳后绘图================================================================================
d_test = d_minMax
k = 3  # k 为要聚类的中心数
purity_all = []
for alpha in range(10,11):
    rs_aver = 0
    purity_aver = 0
    for n in range(10):
        clu_center = init_center(d_test, k)  # 聚类中心
        U = init_U(d_test.shape[0], k)  # 隶属度矩阵
        accuracy = 1
        dij = distance.cdist(d_test, clu_center, 'euclidean')
        times = 0
        while accuracy > 0.0001:
            times += 1
            if times > 1000:
                break
            clu_center_new = clustercenter(U, alpha, d_test, k)
            U = lishu(d_test, clu_center, alpha, k)
            accuracy = np.linalg.norm(
                clu_center_new[clu_center_new[:, 0].argsort()]-clu_center[clu_center[:, 0].argsort()])
            clu_center = clu_center_new

        clu_center = np.array(clu_center)
        kclass = np.argmax(U, axis=1)
        rs = metrics.rand_score(data[:, 4], kclass)
        A = data[kclass == 0]
        B = data[kclass == 1]
        C = data[kclass == 2]
        A_target = Counter(A[:, 4]).most_common(1)[0][0]
        # print(A_target)
        B_target = Counter(B[:, 4]).most_common(1)[0][0]
        # print(B_target)
        C_target = Counter(C[:, 4]).most_common(1)[0][0]
        # print(C_target)
        acc = Counter(A[:, 4]).most_common(1)[0][1] + \
            Counter(B[:, 4]).most_common(1)[0][1] + \
            Counter(C[:, 4]).most_common(1)[0][1]
        purity_aver += acc
        rs_aver += rs
    purity_all.append(purity_aver/1500)
    print('alpha:', alpha, '10次FCM平均兰德指数为:',
          rs_aver/(n+1), '纯度为：', purity_aver/1500)

from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target_names[kclass]
iris_data.head(3).append(iris_data.tail(3))

iris_data.rename(columns={
    "sepal length (cm)": "萼片长",
    "sepal width (cm)": "萼片宽",
    "petal length (cm)": "花瓣长",
    "petal width (cm)": "花瓣宽",
    "species": "种类"},
    inplace=True
)
iris_data.head(3).append(iris_data.tail(3))

kind_dict = {
    "setosa": "类B",
    "versicolor": "类A",
    "virginica": "类C"
}
iris_data["种类"] = iris_data["种类"].map(kind_dict)

sns.pairplot(iris_data, hue='种类')
plt.show()



