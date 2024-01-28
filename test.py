import torch
from torch import nn
from torch.distributions import multinomial
import numpy as np
import random
from d2l import torch as d2l
import matplotlib
# Linear regression
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.001, y.shape) # y加上平均分布的噪声
    return X, y.reshape(num_examples, 1)

test_w = torch.tensor([1, 1.2], dtype=torch.float)
test_b = 10
(X, y) = synthetic_data(test_w, test_b, 10000)
d2l.set_figsize()
d2l.plt.scatter(X[:, (0)].detach().numpy(), y[:, (0)].detach().numpy(), 1) #1表示点的大小
#d2l.plt.show()
d2l.plt.scatter(X[:, (1)].detach().numpy(), y[:, (0)].detach().numpy(),1)
#d2l.plt.show()
indices = list(range(10))
random.shuffle(indices)
print(indices)

board = d2l.ProgressBoard('x')
#for x in np.arange(0, 10, 0.1):
#    board.draw(x, np.sin(x), 'sin', every_n=20)
#    board.draw(x, np.cos(x), 'cos', every_n=10)
#d2l.plt.show()

#raise ValueError("This is a custom error message") #用于手动触发异常
class MyClass:
    def __init__(self, name1, name2):
        self.name1 = name1  # 使用self来存储对象的属性
        self.name2 = name2
    def say_hello(self):
        print(f"Hello, my name is {self.name2}")
# 创建一个MyClass的实例
obj = MyClass("Alice", "wq")
obj.say_hello()

class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
print(data.X[0],data.y[0])
#git push by pycharm test