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
        super(LinRegComposite3, self).__init__()
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

class LinRegRegressor(nn.Module): 
    def __init__(self, n): 
        super(LinRegRegressor, self).__init__() 
        self.linear_regs = nn.ModuleList([nn.Linear(1, 1) for _ in range(n)])
    def forward(self, X): 
        for i, lin_reg in enumerate(self.linear_regs): 
            X = lin_reg(X)
        return X
    def get_weight(self): 
        coef = 1 
        for lin_reg in self.linear_regs: 
            coef *= lin_reg.weight[0][0]
        return coef.item() 
    def get_intercept(self, X): 
        return torch.mean(LinRegRegressor.forward(self, X) - LinRegRegressor.get_weight(self) * X).item()
# %%
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import matplotlib.pyplot as plt 
import numpy as np 
for NOISE_COEF in [0.3, 0.5]: 
    
    N = 1000 
    N_validation = 200
    N_models = 10
    EPOCHS = 10000
    TRIALS = 100
    LR = 0.00005 # 0.001

    mse_train = np.zeros((TRIALS, N_models, EPOCHS))
    mse_val = np.zeros((TRIALS, N_models, EPOCHS))
    meta_differences = np.zeros((TRIALS, N_models, EPOCHS))
    grad_magnitudes = np.zeros((TRIALS, N_models, EPOCHS))

    for trial in range(TRIALS): 
        
        import torch 
        X = torch.randn(N)
        A, B = torch.randn(1), torch.randn(1) 
        noise = torch.randn(N)
        y = A * X + B
        y += noise * NOISE_COEF * torch.mean(y)

        X_val = torch.randn(N_validation)
        noise = torch.randn(N_validation)
        y_val = A * X_val + B 
        y_val += noise * NOISE_COEF * torch.mean(y_val)

        sklearn_linreg = SklearnLinearRegression() 
        sklearn_linreg.fit(X.reshape(-1, 1), y)
        line_of_best_fit_weight = sklearn_linreg.coef_[0]
        line_of_best_fit_intercept = sklearn_linreg.intercept_

        differences = np.zeros((N_models, EPOCHS))
        for i in range(N_models): 
            lin_reg = LinRegRegressor(i + 1)
            mse = nn.MSELoss()
            optimizer = torch.optim.SGD(lin_reg.parameters(), lr=LR)

            
            for epoch in range(EPOCHS): 
                optimizer.zero_grad()
                train_loss = mse(lin_reg(X.reshape(-1, 1)), y.reshape(-1, 1)).sum()
                mse_train[trial][i][epoch] = train_loss.detach().numpy()
                train_loss.backward()
                avg_magnitude = []
                for parameter in lin_reg.parameters(): 
                    avg_magnitude += [torch.norm(parameter.grad)]
                grad_magnitudes[trial][i][epoch] = torch.mean(torch.tensor(avg_magnitude))
                optimizer.step()
                differences[i][epoch] = abs(lin_reg.get_weight() - line_of_best_fit_weight) + abs(lin_reg.get_intercept(X.reshape(-1, 1)) - line_of_best_fit_intercept)
                
                with torch.no_grad(): 
                    mse_val[trial][i][epoch] = mse(lin_reg(X_val.reshape(-1, 1)), y_val.reshape(-1, 1)).sum().detach().numpy()
            meta_differences[trial][i] += differences[i]

    np.save(f"differences/differences-noise={NOISE_COEF}-lr={LR}", meta_differences)
    np.save(f"mse/train_mse-noise={NOISE_COEF}-lr={LR}", mse_train)
    np.save(f"mse/val_mse-noise={NOISE_COEF}-lr={LR}", mse_val)
    # np.save(f"grads/magnitudes_noise={NOISE_COEF}", grad_magnitudes)

# %% 

# import numpy as np 
# import matplotlib.pyplot as plt 

# NOISE=0.15

# """meta_differences = np.load(f"differences/differences-noise={NOISE}.npy")
# mse_val = np.load(f"mse/val_mse-noise={NOISE}.npy")
# for i in range(10): 
#     differences = np.mean(meta_differences[:, i, :],axis=0)
#     plt.plot(differences, label=f"{i+1}th regressor")
# plt.legend()

# for i in range(10): 
#     print(np.std(mse_val[:, i, -1]))"""

# grad_magnitudes = np.load(f"grads/magnitudes_noise={NOISE}.npy")
# for i in range(10): 
#     plt.plot(np.mean(grad_magnitudes[:, i, :], axis=0)[5000:], label=f"LNN-{i}")
# plt.legend()
# plt.xlabel("Training Iterations")
# plt.ylabel("Average Gradient Magnitude")
# plt.title("Gradient Magnitudes over time for All Models")
# plt.show()


# # %%
# """
# So we have created some random data with some best line of fit (corrupted by the noise.) 
# We know can train an sklearn model on this data. 
# """
# import matplotlib.pyplot as plt
# dubs = 0
# for _ in range(100): 
#     import torch 
#     X = torch.randn(1000)
#     A, B = torch.randn(1), torch.randn(1) 
#     noise = torch.randn(1000)
#     y = A * X + B + noise * 0.15 * torch.mean(A * X)

#     # plt.scatter(X.detach().numpy(), y.detach().numpy())
#     # plt.show()
    
#     print(f"true A: {A.item()}, true B: {B.item()}")
    
#     from sklearn.linear_model import LinearRegression as SklearnLinearRegression
#     sklearn_linreg = SklearnLinearRegression() 
#     sklearn_linreg.fit(X.reshape(-1, 1), y)
#     print(f"Closed Form -  A: {sklearn_linreg.coef_[0]}, sklearn B: {sklearn_linreg.intercept_}")
#     import matplotlib.pyplot as plt 

#     """first train linear regression normal with GD"""
#     lin_reg = LinReg(inputSize=1, outputSize=1)
#     mse = nn.MSELoss()

#     optimizer = torch.optim.SGD(lin_reg.parameters(), lr=0.001)
#     differences = []
#     for _ in tqdm(range(10000)): 
#         optimizer.zero_grad()
#         mse(lin_reg(X.reshape(-1, 1)), y.reshape(-1, 1)).sum().backward()
#         optimizer.step()
#         differences.append(abs(A - lin_reg.linear.weight[0][0].detach().numpy()) + abs(B - lin_reg.linear.bias.item()))
#     print(f"Gradient Descent - A: {lin_reg.linear.weight[0][0]}, B: {lin_reg.linear.bias.item()}")

#     lin_reg = LinRegComposite(inputSize=1, outputSize=1)
#     mse = nn.MSELoss()

#     optimizer = torch.optim.SGD(lin_reg.parameters(), lr=0.001)
#     differencesComposite = []
#     for _ in tqdm(range(10000)): 
#         optimizer.zero_grad()
#         mse(lin_reg(X.reshape(-1, 1)), y.reshape(-1, 1)).sum().backward()
#         optimizer.step()
#         differencesComposite.append(abs(A - lin_reg.get_weight().detach().numpy()) + abs(B - lin_reg.get_intercept().detach().numpy()))
#     print(f"Gradient Descent Composite - A: {lin_reg.get_weight()}, B: {lin_reg.get_intercept()}")

    
#     lin_reg = LinRegComposite3()
#     mse = nn.MSELoss()

#     optimizer = torch.optim.SGD(lin_reg.parameters(), lr=0.001)
#     differencesComposite3 = []
#     for _ in tqdm(range(10000)): 
#         optimizer.zero_grad()
#         mse(lin_reg(X.reshape(-1, 1)), y.reshape(-1, 1)).sum().backward()
#         optimizer.step()
#         differencesComposite3.append(abs(A - lin_reg.get_weight().detach().numpy()) + abs(B - lin_reg.get_intercept().detach().numpy()))
#     print(f"Gradient Descent Composite - A: {lin_reg.get_weight()}, B: {lin_reg.get_intercept()}")

#     plt.plot(differences, label="regular")
#     plt.plot(differencesComposite, label="2D linreg")
#     plt.plot(differencesComposite3, label="3D linreg")
#     plt.legend()
#     plt.show()
# # %%
# print(dubs)

# %%
