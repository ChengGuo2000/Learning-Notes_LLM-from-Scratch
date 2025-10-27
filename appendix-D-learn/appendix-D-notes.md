# Appendix D Reading Notes

## Learning Rate Warmup
- **Learning rate warmup** can stabilize the training of complex models, which involves gradually increasing the learning rate from a very low initial value (`initial_lr`) to a maximum value specified by the user (`peak_lr`). 
- Starting the training with smaller weight updates decreases the risk of the model encountering large, destabilizing updates at the beginning of its training phase.

## Cosine Decay
- **Cosine decay** modulates the learning rate throughout the training epochs, making it follow a cosine curve after the warmup stage. It reduces the learning rate to nearly zero, mimicking the trajectory of a half-cosine cycle. 
- The gradual learning decrease in cosine decay aims to decelerate the pace at which the model updates its weights.
- This is particularly important because it helps minimize the risk of overshooting the loss minima during the training process, which is essential for ensuring the stability of the training during its later phases.