# Appendix E Reading Notes

## Low-Rank Adaptation (LoRA)
- **LoRA** is one of the most widely used techniques for **parameter-efficient fine-tuning**, which is a technique that adapts a pretrained model to better suit a specific, often smaller dataset by adjusting only a small subset of the model's weight parameters. 
- The **Low-Rank** aspect refers to the mathematical concept of limiting the model adjustments to a smaller dimensional subspace of the total weight parameter space, which effectively captures the most influential directions of the weight parameter changes during training.
- **LoRA** can be applied to all linear layers in LLM. We focus on a single layer for illustration purposes.
    - Suppose a larget weight matrix **W** is associated with a specific layer. 
    - When training deep NNs, during backprop, we learn a **ΔW** matrix, which contains information on how much we want to update the original weight parameters to minimize the loss function during training. 
    - In regular training and fine-tuning, the weight update is **$W_{updated} = W + ΔW$**.
    - **LoRA** offers a more efficient alternative to computing the weight updates **ΔW** by learning an approximation of it: **$ΔW \approx AB$**, where **A** and **B** are two matrices much smaller than **W**.
    - Using **LoRA**, we can reformulate the weight update we defined earlier: **$W_{updated} = W + AB$**.
    - The inner dimension **r** between **A** and **B** is a tunable hyperparameter.
    - The **distributive law of matrix multiplication** allows us to separate the original and updated weights rather than combine them.
    - The regular fine-tuning with **x** as the input data can be expressed as **$x(W + ΔW) = xW + xΔW$**.
    - For **LoRA**, it is **$x(W + AB) = xW + xAB$**.
- Besides reducing the number of weights to update during training, the ability to keep the **LoRA** weight matrices separate from the original model weights makes **LoRA** even more useful in practice, because it allows the pretrained model weights to remain unchanged, with the **LoRA** matrices being applied dynamically after training when using the model.
- Keeping the **LoRA** weights separate is very useful because it enables model customization without needing to store multiple complete versions of an LLM. This reduces storage requirements and improves scalability, as only the smaller **LoRA** matrices need to be adjusted and saved when we customize LLMs for each specific customer or application.

## Useful Links
- [LoRA Paper](https://arxiv.org/abs/2106.09685)