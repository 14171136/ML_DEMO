# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 20:19
# @Author  : Syao
# @FileName: K_means.py
# @Software: PyCharm

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

class K_means:
    def __init__(self,K,data):
        self.k = K
        self.data = data
        self.initial_cluster()

    def e_distance(self,x,y):
        delta_x = x[0] - y[0]
        delta_y = x[1] - y[1]
        return np.sqrt(np.power(delta_x,2)+np.power(delta_y,2))

    def initial_cluster(self):
        np.random.seed(2020)
        x_max = np.max(np.array(self.data)[:,0])
        x_min = np.min(np.array(self.data)[:,0])
        y_max = np.max(np.array(self.data)[:,1])
        y_min = np.min(np.array(self.data)[:, 1])
        x_r,y_r = x_max-x_min,y_max-y_min
        self.clusters = [[x_min+x_r*np.random.uniform(),y_min+y_r*np.random.uniform()] for i in range(3)]

    def train(self):
        while True:
            self.clusters_dict = {str(i): [] for i in self.clusters}
            cluster_copy = self.clusters.copy()
            for data in self.data:
                index = [self.e_distance(c,data) for c in self.clusters].index(min([self.e_distance(c,data) for c in self.clusters]))
                self.clusters_dict[str(self.clusters[index])].append(list(data))
            else:
                for i in range(3):
                    if self.clusters_dict[str(self.clusters[i])]:
                        new_cluster = list(np.mean(self.clusters_dict[str(self.clusters[i])],axis=0)[:-1])
                        self.clusters[i] = new_cluster
                if self.clusters == cluster_copy:
                    break


if __name__ == '__main__':
    iris_data = load_iris().data[:,2:]
    datas = []
    for i, data in enumerate(iris_data):
        data = np.append(data, load_iris().target[i])
        datas.append(data)

    k_means = K_means(3,datas)
    k_means.train()
    plt.figure()
    color = ['r','b','g']
    c = load_iris().target_names
    print(k_means.clusters_dict)
    for i,(k,points) in enumerate(k_means.clusters_dict.items()):
        x = [points[x][0] for x in range(len(points))]
        y = [points[x][1] for x in range(len(points))]
        plt.scatter(x,y,c=color[i],label=c[i])
    plt.legend()
    plt.show()
