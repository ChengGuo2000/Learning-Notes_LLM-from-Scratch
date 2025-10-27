# Appendix D Reading Notes

## Learning Rate Warmup
- **Learning rate warmup** can stabilize the training of complex models, which involves gradually increasing the learning rate from a very low initial value (`initial_lr`) to a maximum value specified by the user (`peak_lr`). 
- Starting the training with smaller weight updates decreases the risk of the model encountering large, destabilizing updates at the beginning of its training phase.

## Cosine Decay
- **Cosine decay** modulates the learning rate throughout the training epochs, making it follow a cosine curve after the warmup stage. It reduces the learning rate to nearly zero, mimicking the trajectory of a half-cosine cycle. 
- The gradual learning decrease in cosine decay aims to decelerate the pace at which the model updates its weights.
- This is particularly important because it helps minimize the risk of overshooting the loss minima during the training process, which is essential for ensuring the stability of the training during its later phases.

## Gradient Clipping
- **Gradient clipping** enhances the stability during LLM training, which involves setting a threshold above which gradients are downscaled to a predetermined maximum magnitude. 
- This process ensures that the updates to the model's parameters during backpropagation stay within a manageable range.
- Applying the `max_norm = 1.0` setting within PyTorch's `clip_grad_norm_` function ensures that the norm of the gradients does not surpass 1.0. Here, the norm referes to the **L2 norm**, or the **Euclidean norm**.
- Upon calling the `.backward()` method, PyTorch calculates the loss gradients and stores then in a `.grad` attribute for each model weight (parameter) tensor.
- We can identify the highest gradient value by scanning all the `.grad` attributes of the model's weight tensors after calling `.backward()`.