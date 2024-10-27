import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


class DiagonalLinear(nn.Module):
    def __init__(self, input_dim):
        super(DiagonalLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        return x * self.weight + self.bias


class RBFNet(nn.Module):
    def __init__(self, input_dim):
        super(RBFNet, self).__init__()
        self.diagonal_linear = DiagonalLinear(input_dim)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.diagonal_linear(x)
        x = torch.exp(-x**2)  # Gaussian activation
        x = self.linear(x)
        return x


class RBF_fitting:
    # * use machine learning to fit the function
    # * f(x) = w0 + sum_{i=1}^{N} wi * Normal(a_i * x + b_i)
    def __init__(self, x, y, itrs=1000, learning_rate=0.01):
        self.data_x = x
        self.data_y = y
        x = torch.tensor(x).float().reshape(-1, 1)
        y = torch.tensor(y).float().reshape(-1, 1)

        # initialize the model
        rbfnet = RBFNet(input_dim=x.shape[0])

        # define the loss function
        criterion = nn.MSELoss()

        # define the optimizer
        optimizer = torch.optim.Adam(rbfnet.parameters(), lr=learning_rate)

        # train the model
        tqdm_iter = tqdm(range(itrs))
        for epoch in tqdm_iter:
            optimizer.zero_grad()
            output = rbfnet(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            tqdm_iter.set_description(f"Loss: {loss.item()}")

        self.rbfnet = rbfnet

    def predict(self, x):
        return self.rbfnet(x)

    def draw(self, bins=1000):
        # draw the data points
        plt.scatter(self.data_x, self.data_y, label="data")
        # draw the fitting curve
        x_min = min(self.data_x)
        x_max = max(self.data_x)
        x = np.linspace(x_min, x_max, bins).reshape(-1, 1)
        x_input = torch.tensor(x).float()
        plt.plot(x, self.predict(x_input).detach().numpy(), label="fitting")
        plt.legend()
        # show the plot
        plt.show()


if __name__ == "__main__":
    # generate the data
    x = [1, 2, 3, 4, 5]
    y = [10, 0, 10, 0, 10]

    # fit the data
    rbf_fitting = RBF_fitting(x, y, itrs=10000, learning_rate=0.01)

    # draw the fitting curve
    rbf_fitting.draw()
