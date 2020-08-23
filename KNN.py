# -*- coding: utf-8 -*-
# @Time    : 2020/8/23 11:11
# @Author  : Syao
# @FileName: KNN.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(2020)

def creat_dataset(show=False):
    data1 = np.random.random([20,1,2])
    label1 = np.reshape(np.array([1]*20),[20,1,1])
    data = np.dstack([data1,label1])

    data2 = np.random.random([20,1,2])+1
    label2 = np.reshape(np.array([0]*20),[20,1,1])
    data = np.concatenate([data,np.dstack([data2,label2])],axis=0)

    np.random.shuffle(data)
    if show:
        plt.scatter(data[:, :, 0], data[:, :, 1])
        plt.show()
    return data

def e_distance(x,y):
    return np.sqrt(np.sum(np.power(x-y,2)))

def knn(data,test,k):

    dis = []
    for i,train_data in enumerate(data):
        d = e_distance(test,train_data[:,:2][0])
        dis.append([d,train_data[:,-1][0]])
    print(dis)
    dis = sorted(dis,key=lambda x:x[0])[:k]
    labels = [x[1] for x in dis]
    label = max(labels,key=labels.count)
    plt.figure()
    plt.scatter(data[data[:,:,-1]==0][:,0],data[data[:,:,-1]==0][:, 1],c='r',label='Label 0')
    plt.scatter(data[data[:, :, -1] == 1][:, 0], data[data[:, :, -1] == 1][:, 1], c='g',label='Label 1')
    plt.scatter(test[:,:,0],test[:,:,1],c='r' if label==0 else 'g',marker='^',label='Test',s=60)
    plt.legend()
    plt.show()
    return label

def main():
    test = np.random.random([1,1,2])+np.random.randint(0,3)
    data = creat_dataset()
    label = knn(data,test,10)
    print(label)

if __name__ == '__main__':
    main()








