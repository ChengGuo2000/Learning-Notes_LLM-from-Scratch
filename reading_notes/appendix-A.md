# Appendix A Reading Notes

## PyTorch Introduction
- Three Components
    - **Tensor library**: extends NumPy, accelerates computation on GPUs, a fundamental building block for computing.
    - **Automatic differentiation engine**: AKA autograd, automatic computation of gradients from tensor operations, simplifying backprop and model optimization.
    - **DL library**: offers modular, flexible, and efficient building blocks, including pretrained models, loss functions, and optimizers.
- **Torch** in **PyTorch** acknowledges the library's roots in **Torch**, a scientific computing framework with wide support for ML algorithms, initially created using Lua programming language.

## Tensor
- Tensor is a mathematical concept that generalizes vectors and matrices to potentially higher dimensions, can be characterized by their order (or rank), which is the number of dimensions. 
    - **Scalar**: a tensor of rank 0 (0D tensor).
    - **Vector**: a tensor of rank 1 (1D tensor).
    - **Matrix**: a tensor of rank 2 (2D tensor).
- Tensors serve as data containers, hold multidimensional data, each dimension represents a different feature.
- A 32-bit floating-point number offers sufficient precision for most DL tasks while consuming less memory and computational resources. GPU architectures are optimized for 32-bit computations.
- Reshaping tensors
    - **.view()** requires the original data to be continguous and will fail if it isn't.
    - **.reshape()** will work regardless, copying the data if necessary to ensure the desired shape.
- **.T** is for transposing a tensor, fillping it across its diagonal.
- Matrix Multiplication with **.matmul()** or the **@** operator.

## Automatic Differentiation (autograd)
- **Computational graph**: a directed graph that allows us to express and visualize mathematical expressions. In DL, it lays out the sequence of calculations needed to compute the output of a NN, and computes the reqired gradients for backprop.
- If we carry out computations in PyTorch, it iwll build a computational graph internally by default if one of its terminal nodes has the `requires_grad` attribute set to **True**.
- **Gradient**: a vector containing all of the partial derivatives of a multivariate function.
- The `.backward()` method will let PyTorch compute the gradients of all leaf nodes in the graph, which will be stored via the tensors' `.grad` attributes.

## Implementing NN
- Subclass the `torch.nn.Module` class to define custon NN. The `Module` base class has lots of functionality, including encapsulate layers and operations and keep track of the model's parameters.
- Define the network layers in the `__init__` constructor and specify how the layers interact in the forward method.
- The forward method describes how the input data passes through the network and comes together as a computation graph.
- The backward method are not needed to implement by ourselves.
- `torch.nn.Linear` layers are the **feedforward** or **fully connected** layers.
- Each parameter for which `requires_grad=True` counts as a trainable parameter and will be updated during training, which is default for weights and biases in `torch.nn.Linear`.
- Initializing model weights with small random numbers is desired to *break symmetry* during training.
- `grad_fn=<AddmmBackward0>` means that the tensor was created via a matrix multiplication and addition operation.
- **Addmm** stands for matrix multiplication (**mm**) followed by an addition (**Add**).
- `torch.no_grad()` is used when we use the network without training or backprop, when we use it for prediction and inference after training.
- Commonly, we return the outputs of the last layer (**logits**) without passing them to a nonlinear activation function.
- PyTorch's commonly used loss functions combine the **softmax** (or **sigmoid** for binary classification) operation with the negative log-likelihood loss in a single class for numerical efficiency and stability.


## Dataset and DataLoader
- A custom `Dataset` class contains the three main components.
    - `__init__`: setting up attributes to access later (file paths, file objects, database connectors).
    - `__getitem__`: returning exactly one item from the dataset via an `index`, which is the features and class label for a single instance.
    - `__len__`: retrieving the length of the dataset.
- A **training epoch** is when the `train_loader`iterates over the training dataset, visiting each training example exactly once.
- Having a substantially smaller batch as the last batch can disturb the convergence, so to prevent this, we set `drop_last=True`.
- `num_workers` in `DataLoader` is crucial for parallelizing data loading and preprocessing.
    - **Set to 0**: data loading will be done in the main process and not in separate worker processes. This may lead to significant slowdowns
    - **Set to a number > 0**: multiple worker processes are launched to load data in parallel, freeing the main process to focus on training, better utilizing resources. But it is not optimal when working with very small datasets or on Jupyter notebooks because it may lead to overhead or crashes.
    - `num_workers=4` usually leads to optimal performance in real-world datasets.

## Useful Links
- [PyTorch Website](https://pytorch.org/)
- [Papers With Code - Trends](https://paperswithcode.com/trends)
- [Kapple Data Science and Machine LEarning Survey 2022](https://www.kaggle.com/c/kaggle-survey-2022)
- [Google Colab](https://colab.research.google.com)
- [Supplementary Code for Exercise A.2](https://mng.bz/o05v)
- [Scientific Computing in Python: Introduction to NumPy and Matplotlib](https://sebastianraschka.com/blog/2020/numpy-intro.html)
- [PyTorch's Official Documentation on different tensor data types](https://docs.pytorch.org/docs/stable/tensors.html)