# -*- coding: utf-8 -*-
# @Time    : 2020/8/21 10:37
# @Author  : Syao
# @FileName: linear_reg.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
class Linear_regression:
    def __init__(self,x,y,lr,epoch):
        self.x = x
        self.y = y
        self.n = len(self.x)
        self.lr = lr
        self.epoch = epoch
        self._initial_params()

    def _initial_params(self):
        self.a = np.random.uniform()
        self.b = np.random.uniform()

    def calculate_y(self):
        y_hat = self.a * x + self.b
        return y_hat

    def calculate_loss(self):
        return np.mean(np.sum(np.square(self.calculate_y()-self.y)))

    def calculate_gradient(self):
        delta_a = 2 * np.mean((self.a*self.x +self.b - self.y)*self.x)
        delta_b = 2*np.mean(self.a*self.x +self.b - self.y)
        return delta_a,delta_b

    def train(self):
        for i in range(self.epoch):
            theta_a,theta_b = self.calculate_gradient()
            self.a -= self.lr*theta_a
            self.b -= self.lr*theta_b
            print('Epoch {0}=====>MSE loss:{1:.4f}'.format(i+1,self.calculate_loss()))


if __name__ == '__main__':
    x = np.arange(1, 20, 1) + 0.01 * np.random.uniform(-1, 1)
    y = x * 2 + 4

    l_reg = Linear_regression(x,y,0.001,1000)
    l_reg.train()
    print(l_reg.a,l_reg.b)
    plt.figure()
    plt.plot(np.linspace(1,20),l_reg.a*np.linspace(1,20)+l_reg.b,c='b')
    plt.scatter(x,y,c='r')
    plt.show()



