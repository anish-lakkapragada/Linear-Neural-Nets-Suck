import torch 
from torch import nn as nn 
import sklearn 
from sklearn import datasets, model_selection
import numpy as np
import argparse
import matplotlib.pyplot as plt 
from utils import alternate_loss_function, LogisticRegression

""""
get the mean and standard deviation of performance across K=3 folds 

> for each fold
> train the logistic regression model for 100 iterations with log loss
> train the alternate logistic regression model 
"""
parser = argparse.ArgumentParser() 
parser.add_argument('--splits')
parser.add_argument('--lr')
parser.add_argument('--epochs')
args = parser.parse_args()

X, y = datasets.load_breast_cancer(return_X_y=True)
X, y = sklearn.utils.shuffle(X, y, random_state=42)
skfolds = model_selection.StratifiedKFold(n_splits=int(args.splits))


for fold, (train_index, test_index) in enumerate(skfolds.split(X, y)):
    X_train, y_train, X_test, y_test = map(lambda list: torch.tensor(list).type(torch.float32), [X[train_index], y[train_index], X[test_index], y[test_index]])
    log_reg = LogisticRegression(input_dim=30, output_dim=1)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(log_reg.parameters(), lr=float(args.lr))
    log_reg_losses = []
    for epoch in range(int(args.epochs)): 
        log_reg.apply_sigmoid = False
        # optimizer.zero_grad() 
        # loss = criterion(torch.squeeze(log_reg(X_train)), y_train)
        # loss.backward()
        # with torch.no_grad(): log_reg_losses.append(loss)
        # optimizer.step()

        log_reg.apply_sigmoid = True 
        optimizer.zero_grad()
        loss = alternate_loss_function(torch.squeeze(log_reg(X_train)), y_train)
        loss.backward()
        with torch.no_grad(): log_reg_losses.append(loss)
        optimizer.step() # change parameters
        print(loss)
    plt.plot(torch.tensor(log_reg_losses).detach().numpy())
    plt.show()