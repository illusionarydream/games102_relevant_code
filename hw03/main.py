import numpy as np
import matplotlib.pyplot as plt
from RBF import RBFNet
import torch
import torch.nn as nn
import torch.optim as optim


class Curve_fitting:
    def __init__(self, learning_rate=0.01):
        self.data_x = np.array([])
        self.data_y = np.array([])
        self.t = None
        # for training
        self.learning_rate = 0.01
        self.epochs = 1000
        # for drawing
        self.xlim = (0, 1.0)
        self.ylim = (0, 1.0)

    def add_point(self, x, y):
        # add a point to the dataset
        self.data_x = np.append(self.data_x, x)
        self.data_y = np.append(self.data_y, y)
        # parameterize the curve, initialize the t parameter
        self.parameterize()
        # initialize the t-x model and t-y model
        self.t_x_model = RBFNet(input_dim=self.t.shape[0])
        self.t_y_model = RBFNet(input_dim=self.t.shape[0])
        # train the t-x model and t-y model
        self.train()

    def parameterize(self, method='centripetal'):
        # parameterize the curve
        if method == 'uniform':
            self.t = np.linspace(0, 1, len(self.data_x))

        if method == 'chordal':
            self.t = np.zeros(len(self.data_x))
            for i in range(1, len(self.data_x)):
                self.t[i] = self.t[i-1] + \
                    np.linalg.norm(self.data_x[i] - self.data_x[i-1])
            self.t /= self.t[-1]

        if method == 'centripetal':
            self.t = np.zeros(len(self.data_x))
            for i in range(1, len(self.data_x)):
                self.t[i] = self.t[i-1] + \
                    np.sqrt(np.linalg.norm(self.data_x[i] - self.data_x[i-1]))
            self.t /= self.t[-1]

    def train(self):
        # set the loss function and optimizer
        criterion = nn.MSELoss()
        # train the t-x model and t-y model
        t = torch.tensor(self.t).float().reshape(-1, 1)
        x = torch.tensor(self.data_x).float().reshape(-1, 1)
        y = torch.tensor(self.data_y).float().reshape(-1, 1)
        # iterate to train the model
        # x model
        optimizer_x = torch.optim.Adam(
            self.t_x_model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            optimizer_x.zero_grad()
            output_x = self.t_x_model(t)
            loss = criterion(output_x, x)
            loss.backward()
            optimizer_x.step()
        # y model
        optimizer_y = torch.optim.Adam(
            self.t_y_model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            optimizer_y.zero_grad()
            output_y = self.t_y_model(t)
            loss = criterion(output_y, y)
            loss.backward()
            optimizer_y.step()

    def predict(self, t):
        # predict the x and y values of the curve at the given t
        t = torch.tensor(t).float().reshape(-1, 1)
        x = self.t_x_model(t)
        y = self.t_y_model(t)
        return x.detach().numpy(), y.detach().numpy()

    def plot_curve(self):
        # plot the curve
        plot_t = np.linspace(0, 1, 1000)
        plot_x, plot_y = self.predict(plot_t)
        # clear the plot
        plt.clf()
        plt.plot(self.data_x, self.data_y, 'bo-', label='Curve')
        plt.plot(plot_x, plot_y, 'r-', label='Fitted Curve')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Interactive Curve Fitting')
        plt.legend()
        if self.xlim and self.ylim:
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
        plt.draw()


# 实例化 Curve_fitting 类
curve_fitting = Curve_fitting()

# 定义鼠标点击事件处理函数


def onclick(event):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        curve_fitting.add_point(x, y)
        curve_fitting.plot_curve()


# 创建图形和轴
fig, ax = plt.subplots()
ax.set_title('Click to add points')
fig.canvas.mpl_connect('button_press_event', onclick)

# 显示图形
plt.show()
