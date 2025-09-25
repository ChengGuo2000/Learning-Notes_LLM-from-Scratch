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

## Shortcut Connections
- **Shortcut Connections**, also called **skip** or **residual** **connections** is a technique that is utilized in **ResNet** to overcome the limitations posed by the vanishing gradient problem in deep NN.
- **Shortcut Connections** creates an alternative, shorter path for the gradient to flow through the network by skipping one or more layers, which is achieved by adding the output of one layer to the output of a later layer. 
- `.backward()` method is a convenient method in PyTorch that computes loss gradients, which are required during model training, without implementing the math for the gradient calculation ourselves, thereby making working with deep NN much more accessible.

## Build a transformer block
- The **self-attention mechanism** in the **multi-head attention** block identifies and analyzes relationships between elements in the input sequence. In contrast, the **feed forward network** modifies the data individually at each position. This combination not only enables a more nuanced understanding and processing of the input but also enhances the model's overall capacity for handling complex data patterns.
- **Layer Normalization** is applied before each of these two components, and **dropout** is applied after them to regularize the model and prevent overfitting. This is also known as **Pre-LayerNorm**. 
- Older architectures, such as the original transformer model, applied **layer normalization** after the self-attention and feed forward netwworks instead, known as **Post-LayerNorm**, which often leads to worse training dynamics.
- The **preservation of shape** throughout the transformer block architecture is not incidental but a crucial aspect of its design. This design enables its effective application across a wide range of sequence-to-sequence tasks, where each output vector directly corresponds to an input vector, maintaining a one-to-one relationship.
- The output is a context vector that encapsulates information from the entire input sequence, which means while the physical dimension remain unchanged, the content of each output vector is re-encoded to integrate contextual information from across the entire input sequence.

## Useful Links
- [Language Models Are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Lambda Labs](https://lambda.ai/)