# -*- coding: utf-8 -*-
# @Time    : 2020/8/23 16:11
# @Author  : Syao
# @FileName: Bayes.py
# @Software: PyCharm

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def creat_dataset():
    datas_pd = load_iris()
    datas_x = datas_pd.data
    datas_y = datas_pd.target[:,np.newaxis]
    x_train, x_test, y_train, y_test = train_test_split(datas_x, datas_y, test_size=0.2, random_state=2020)
    return x_train,x_test,y_train,y_test,list(set(datas_pd.target))

class Bayes:
    def __init__(self,x_train,y_train,x_test,y_test,labels):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.labels = labels

    def fit(self):
        self.pred = []
        for x_test in self.x_test:
            class_res_dict = {}
            for c in self.labels:
                c_data = np.hstack([self.x_train,self.y_train])
                c_data = c_data[c_data[:,-1]==c]
                p_c = sum(self.y_test==c) / len(self.y_test)
                for i,feature in enumerate(x_test):
                    p_feature_c = len(c_data[c_data[:,i]==feature]) / len(c_data)
                    p_c *= p_feature_c
                class_res_dict[c] = p_c
            self.pred.append(max(class_res_dict,key=class_res_dict.get))
        self.pred = np.array(self.pred)

    def evaluate(self):
        hit = 0
        for y,y_pred in zip(self.y_test,self.pred):
            if y==y_pred:
                hit += 1
        print('Accuracy: {0}'.format(hit/len(self.y_test)))

if __name__ == '__main__':
    x_train, x_test, y_train, y_test, labels = creat_dataset()
    bayes = Bayes(x_train, y_train,x_test, y_test, labels)
    bayes.fit()
    bayes.evaluate()
    print(bayes.pred,'\n',bayes.y_test.ravel())
