# Chapter 4 Reading Notes

## Layer Normalization
- **Layer Normalization** is to adjust the activations (outputs) of a NN layer to have a mean of 0 and a variance of 1, also known as **unit variance**. It speeds up the convergence to effective weights and ensures consistent, reliable training. In modern transformers, it is typically applied before and after the multi-head attention module and before the final output layer.
- `dim = 0` calculates mean across the row dimension to obtain one mean per column. 
- `dim = 1` or `dim = -1` calculates mean across the column dimension to obtain one mean per row.
- When calculating variance, setting `unbiased = False` prevents Bessel's correction, which uses n - 1 instead of n as the denominator. For LLMs, this difference is practically negligible, but to ensure compatibility with GPT-2 model's normalization layers and reflecting TensorFlow's default behavior, we are going to set it to **False**.
- **Batch normalization** normalizes across the batch dimension. **Layer normalization** normalizes across the feature dimension, and it normalizes each input independently of the batch size, which offers more flexibility and stability when training LLMs with significant computational resources, so it is particularly beneficial for distributed training or when deploying models in environments where resources are constrained. 

## GELU activation function
- **GELU** (Gaussian error linear unit) and **SwiGLU** (Swish-gated linear unit) are more complex and smooth activation functions incorporating Gaussian and sigmoid-gated linear units, so they offer improved performance for DL models. 
- For **GELU**, instead of implementing the exact version, it is common to implement a computationally cheaper approximation, which is also used in GPT-2 model. 
- **GELU** allows for a small, non-zero output for negative values, so neurons that receive negative input can still contribute to the learning process, and it also offers a non-zero gradient for almost all negative values.

## Useful Links
- [Language Models Are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Lambda Labs](https://lambda.ai/)