import numpy as np
import torch 

def alternate_loss_function(y_hat, y): 
    return ((y_hat + y - 2 * y * y_hat) ** 2).sum()

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)
         self.apply_sigmoid = None
     def forward(self, x):
         outputs = self.linear(x)
         if self.apply_sigmoid: outputs = torch.sigmoid(outputs)
         return outputs