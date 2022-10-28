# %% 
import torch 
from torch import nn as nn 
from tqdm import tqdm 


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
    
# %% 
import torch 
X = torch.randn(100)
A, B = torch.randn(1), torch.randn(1) 
noise = torch.randn(100)
y = A * X + B + noise * 0.2
print(f"true A: {A.item()}, true B: {B.item()}")
# %%
"""
So we have created some random data with some best line of fit (corrupted by the noise.) 
We know can train an sklearn model on this data. 
"""

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
sklearn_linreg = SklearnLinearRegression() 
sklearn_linreg.fit(X.reshape(-1, 1), y)
print(f"Closed Form -  A: {sklearn_linreg.coef_[0]}, sklearn B: {sklearn_linreg.intercept_}")
# %%
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
# %%
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
# %%
plt.plot(differences[-10:])
plt.plot(differencesComposite[-10:], color="red")
# %%
