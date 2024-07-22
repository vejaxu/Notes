### torch.unsqueeze()

turns an n.d. tensor into an (n+1).d. one by adding an extra dimension of depth 1.

![](D:\Code\Notes\learnpytorch\torch.unsqueeze.png) 

### plot.figure(figsize=(10, 7))

This will modify/change the width and height of the plot.

### plt.scatter()

plt.scatter(x, y, s = None, c = None)

s denotes float or array-like, shape(n, )

c denotes color

### torch.randn(size)

generate a tensor with random values drawn from a normal distribution with mean 0 and variance 1 (standard normal distribution)

```python
# Generate a single random value
random_value = torch.randn(1)
print(random_value)

# Generate a 1D tensor with 5 random values
random_tensor_1d = torch.randn(5)
print(random_tensor_1d)
```

### PyTorch model building essentials

[`torch.nn`](https://pytorch.org/docs/stable/nn.html), [`torch.optim`](https://pytorch.org/docs/stable/optim.html), [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html)

### model.parameters()

check model's parameters using .parameters()

We can also get the state (what the model contains) of the model using .state_dict()

### loss function and optimizer

For our model to update its parameters on its own, we'll need to add a few more things to our recipe.

And that's a **loss function** as well as an **optimizer**.

Measures how wrong your models predictions (e.g. `y_preds`) are compared to the truth labels (e.g. `y_test`). Lower the better.

Mean absolute error (MAE) for regression problems ([`torch.nn.L1Loss()`](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)). 

Binary cross entropy for binary classification problems ([`torch.nn.BCELoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)).

Tells your model how to update its internal parameters to best lower the loss.

Stochastic gradient descent ([`torch.optim.SGD()`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)). 

Adam optimizer ([`torch.optim.Adam()`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam))

### Training loop

It's train time !

do the forward pass, 

calculate the loss,

optimizer zero grad,

losssss backwards !



Optimizer step step step 



Let's test now !

with torch no grad:	

do the forward pass,

calculate the loss,

watch it go down down down !

![](D:\Code\Notes\learnpytorch\training loop.png)

### Testing loop

![](D:\Code\Notes\learnpytorch\testing loop.png)

### loss.detach().numpy()

In PyTorch, when you compute a loss during training, it's typically a tensor. If you want to convert this loss tensor to a NumPy array, you need to detach it from the computation graph and then convert it to a NumPy array.

