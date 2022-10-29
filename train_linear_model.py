# %% 
import torch 
from torch import nn as nn 
from tqdm import tqdm 
import numpy as np

class LinReg(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinReg, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
    
class LinRegComposite(nn.Module): 
    def __init__(self, inputSize, outputSize):
        super(LinRegComposite, self).__init__()
        self.linear= torch.nn.Linear(inputSize, outputSize)
        self.linear2 = torch.nn.Linear(outputSize, outputSize)
    def forward(self,x):
        out = self.linear(x)
        return self.linear2(out)
    def get_weight(self): 
        return self.linear.weight[0][0] * self.linear2.weight[0][0]
    def get_intercept(self): 
        return self.linear2.weight[0][0] * self.linear.bias.item() + self.linear2.bias.item()

class LinRegComposite3(nn.Module): 
    def __init__(self): 
        self.linear = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(1, 1) 
        self.linear3 = torch.nn.Linear(1, 1)
    def forward(self, x): 
        return self.linear3(self.linear2(self.linear(x)))
    def get_weight(self): 
        # m3(m2(m1x + b1) + b2) + b3
        #m3m2m1
        return self.linear.weight[0][0] * self.linear2.weight[0][0] * self.linear3.weight[0][0]
    def get_intercept(self): 
        # m3(m2b1 + b2) + b3
        #m3m2b1 + m3b2 + b3
        m3 = self.linear3.weight[0][0]
        m2 = self.linear2.weight[0][0]
        m1 = self.linear.weight[0][0]
        b1 = self.linear.bias.item()
        b2 = self.linear2.bias.item()
        b3 = self.linear3.bias.item()
        return m3 * m2 * b1 + m3 * b2 + b3
    
# %%
"""
So we have created some random data with some best line of fit (corrupted by the noise.) 
We know can train an sklearn model on this data. 
"""
import matplotlib.pyplot as plt
dubs = 0
for _ in range(100): 
    import torch 
    X = torch.randn(1000)
    A, B = torch.randn(1), torch.randn(1) 
    noise = torch.randn(1000)
    y = A * X + B + noise * 0.15 * torch.mean(A * X)

    # plt.scatter(X.detach().numpy(), y.detach().numpy())
    # plt.show()
    
    print(f"true A: {A.item()}, true B: {B.item()}")
    
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression
    sklearn_linreg = SklearnLinearRegression() 
    sklearn_linreg.fit(X.reshape(-1, 1), y)
    print(f"Closed Form -  A: {sklearn_linreg.coef_[0]}, sklearn B: {sklearn_linreg.intercept_}")
    import matplotlib.pyplot as plt 

    """first train linear regression normal with GD"""
    lin_reg = LinReg(inputSize=1, outputSize=1)
    mse = nn.MSELoss()

    optimizer = torch.optim.SGD(lin_reg.parameters(), lr=0.001)
    differences = []
    for _ in tqdm(range(10000)): 
        optimizer.zero_grad()
        mse(lin_reg(X.reshape(-1, 1)), y.reshape(-1, 1)).sum().backward()
        optimizer.step()
        differences.append(abs(A - lin_reg.linear.weight[0][0].detach().numpy()) + abs(B - lin_reg.linear.bias.item()))
    print(f"Gradient Descent - A: {lin_reg.linear.weight[0][0]}, B: {lin_reg.linear.bias.item()}")

    lin_reg = LinRegComposite(inputSize=1, outputSize=1)
    mse = nn.MSELoss()

    optimizer = torch.optim.SGD(lin_reg.parameters(), lr=0.001)
    differencesComposite = []
    for _ in tqdm(range(10000)): 
        optimizer.zero_grad()
        mse(lin_reg(X.reshape(-1, 1)), y.reshape(-1, 1)).sum().backward()
        optimizer.step()
        differencesComposite.append(abs(A - lin_reg.get_weight().detach().numpy()) + abs(B - lin_reg.get_intercept().detach().numpy()))
    print(f"Gradient Descent Composite - A: {lin_reg.get_weight()}, B: {lin_reg.get_intercept()}")
    # plt.plot(differences)
    # plt.plot(differencesComposite, color="red")
    # plt.plot(np.array(differencesComposite) - np.array(differences))
    if (differencesComposite[0] - differences[0] > 0): 
        print("success")
        dubs += 1
    else: 
        print("no success")
    # plt.show()# %%

# %%
print(dubs)
