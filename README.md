# When More Parameters Reduce Training Performance: Linear Neural Networks

Code for proving that neural networks without activations, or Linear Neural Networks (LNNs), despite being the same as neural networks are actually harder to optimize due to their excess of parameters. This is because having more parameters leads to updates for parameters being determined by other currently suboptimal parameters in iterative optimization methods. Thus, there is a nonconvex objective function which impirically leads to local minimas in optimization; the paper demonstrates this empirically as well. 

`train_linear_model.py` provides all code required for the experiments used. 

