# -*- coding: utf-8 -*-
# @Time    : 2020/8/23 19:26
# @Author  : Syao
# @FileName: Random_Forest.py
# @Software: PyCharm
import numpy as np
import pandas as pd

class Random_Forest:
    def __init__(self,filename):
        self.filename = filename
        self.dataset = self.load_csv()
        self.convert_str_to_float()


    def load_csv(self):
        dataset = []
        with open(self.filename,'r') as f:
            datas = f.readlines()
            for data in datas:
                data = data.strip().split(',')
                dataset.append(data)
        return dataset

    def convert_str_to_float(self):
        for data in self.dataset:
            for col in range(len(self.dataset[0])-1):
                data[col] = float(data[col].strip())

    def splitDataset(self,dataset,k_folds):
        fold_size = int(len(dataset)/k_folds)
        datas = []
        for i in range(k_folds):
            dataset_copy = dataset.copy()
            fold = []
            while len(fold) < fold_size:
                index = np.random.randint(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            datas.append(fold)
        return datas

    def test_split(self,index,value,dataset):
        left,right = [],[]
        for data in dataset:
            if data[index] < value:
                left.append(data)
            else:
                right.append(data)
        return left,right

    def gini_index(self,groups,classes):
        gini = 0
        for c in classes:
            for group in groups:
                size = len(group)
                if size == 0:
                    continue
                p = [row[-1] for row in group].count(c) / size
                gini += p * (1-p)
        return gini

    def get_split(self,dataset,n_features):
        classes = list(set(row[-1] for row in dataset))
        features = []
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        while len(features) < n_features:
            index = np.random.randint(0,len(dataset[0])-1)
            if index not in features:
                features.append(index)
        for index in features:
            for data in dataset:
                groups = self.test_split(index,data[index],dataset)
                gini = self.gini_index(groups,classes)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, data[index], gini, groups

        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def to_terminal(self,group):
        o = [row[-1] for row in group]
        return max(set(o),key=o.count)

    def split(self,node,max_depth,min_size,n_features,depth):
        l,r = node['groups']
        del(node['groups'])
        if not l or r:
            node['left'] = node['right'] = self.to_terminal(l+r)
            return

        if depth >= max_depth:
            node['left'],node['right'] = self.to_terminal(l),self.to_terminal(r)
            return

        if len(l) <= min_size:
            node['left'] = self.to_terminal(l)
        else:
            node['left'] = self.get_split(l,n_features)
            self.split(node['left'],max_depth,min_size,n_features,depth+1)

        if len(r) <= min_size:
            node['right'] = self.to_terminal(r)
        else:
            node['right'] = self.get_split(r,n_features)
            self.split(node['right'],max_depth,min_size,n_features,depth+1)

    def build_tree(self,dataset,max_depth,n_feature,min_size):
        root =  self.get_split(dataset,n_feature)
        self.split(root,max_depth,min_size,n_feature,1)
        return root

    def predict(self,node,data):
        if data[node['index']] < node['value']:
            if isinstance(node['left'],dict):
                return self.predict(node['left'],data)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], data)
            else:
                return node['right']

    def subsample(self,dataset,ratio):
        sample = []
        n = int(len(dataset) * ratio)
        while len(sample) < n:
            index = np.random.randint(len(dataset))
            sample.append(dataset[index])
        return sample

    def bagging(self,trees,data):
        prediction = [self.predict(tree,data) for tree in trees]
        return max(set(prediction),key=prediction.count)

    def random_forest(self,train,test,max_depth,min_size,sample_size,n_trees,n_features):
        trees = []
        for i in range(n_trees):
            sample = self.subsample(train,sample_size)
            tree = self.build_tree(train,max_depth,n_features,min_size)
            trees.append(tree)
        pred = [self.bagging(trees,row) for row in test]
        return pred

    def accuarcy(self,pred,true):
        hit = 0
        for i in range(len(true)):
            if pred[i] == true[i]:
                hit += 1
        return hit / len(true)

if __name__ == "__main__":
    RF = Random_Forest('sonar.all-data.csv')
    N_FOLDS = 5
    folds = RF.splitDataset(RF.dataset,N_FOLDS)
    print(folds[0])
    scores = []
    for fold in folds:
        trainset = folds[:]
        trainset.remove(fold)
        trainset = sum(trainset,[])
        testset = []
        for test_data in fold:
            test_data_copy = test_data.copy()
            test_data_copy[-1] = None
            testset.append(test_data_copy)
        actual = [row[-1] for row in fold]
        pred = RF.random_forest(trainset,testset,max_depth=20,min_size=5,
                                sample_size=0.8,n_trees=5,n_features=15)
        accuracy = RF.accuarcy(pred,actual)
        scores.append(accuracy)
    print('Mean Accuarcy:{}'.format(sum(scores)/len(scores)))
