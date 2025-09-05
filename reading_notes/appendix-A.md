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

## Useful Links
- [PyTorch Website](https://pytorch.org/)
- [Papers With Code - Trends](https://paperswithcode.com/trends)
- [Kapple Data Science and Machine LEarning Survey 2022](https://www.kaggle.com/c/kaggle-survey-2022)
- [Google Colab](https://colab.research.google.com)
- [Supplementary Code for Exercise A.2](https://mng.bz/o05v)
- [Scientific Computing in Python: Introduction to NumPy and Matplotlib](https://sebastianraschka.com/blog/2020/numpy-intro.html)
- [PyTorch's Official Documentation on different tensor data types](https://docs.pytorch.org/docs/stable/tensors.html)