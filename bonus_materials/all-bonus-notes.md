# Bonus Material Reading Notes

## Byte Pair Encoding (BPE)
- **BPE** is used in GPT-2 to GPT-4 and Llama-3. The main idea of **BPE** is to convert text into an integer representation (token IDs) for LLM training.
- A byte consists of 8 bits, so there are otal 256 possible values that a single byte can represent, ranging from 0 to 255. A **BPE** tokenizer usually uses these 256 values as its first 256 single-character tokens.
- The goal of BPE is to build a vocabulary of commonly occurring subwords like `298: ent` and complete words.
- **BPE** algorithm outline:
    - **Identify frequent pairs**: in each iteration, scan the text to find the most commonly occurring pair of bytes
    - **Replace and record**: replace that pair with a new placeholder ID, record this mapping in a lookup table, the size of this lookup table is the "vocabulary size"
    - **Repeat until no gains**: keep repeating steps 1 and 2, continually merging the most frequent pairs, stop when no further compression is possible (no pairs occurs more than once).
    - **Decompression (decoding)**: to retore the original text, reverse the process by substituting each ID with its corresponding pair, using the lookup table.
- One Example `the car in the hat`:
    - Iteration 1: `th` appears twice, so replace it with a new token ID 256, `256: "th`.
    - Iteration 2: `<256>e` appears twice, so replace it with 257, `257: <256>e`.
    - Iteration 3: `<257> ` appears twice, so replace it with 258, `258: <257> `.
    - Now the text become `<258>cat in <258>hat`.

## PyTorch Buffers
- **PyTorch Buffers** are tensor attributes associated with a PyTorch module or model similar to parameters. They are particularly useful when dealing with GPU computations, as they need to be transferred between devices (from CPU to GPU) alongside the model's parameters.
- Unlike parameters, buffers are not updated during training, and they do not require gradient computation, but they still need to be on the correct device to ensure that all computations are performed correctly. 
- We can use PyTorch buffers via `self.register_buffer`.
- Another advantage of PyTorch buffers, over regular tensors, is that they get included in a model's `state_dict`, which is useful when saving and loading trained PyTorch models. 

## KV Cache
- A **KV Cache** stores intermediate key (K) and value (V) computations for reuse during inference, which results in a substantial spped-up when generating responses. The downside is that it adds some complexity to the code, increases memory usage, and can't be used during training. However, the inference speed-ups are often wel worth the trade-offs in code complexity and memory when deploying LLMs.
- Imagine the LLM is given a prompt "Time flies". It generate one token at a time, say it is "fast", so the prompt for the next round is "Time flies fast", the keys and value vectors for the first two tokens are exactly the same, and it would be wasteful to recompute them in each next-token text generation round. So, the idea of the KV cache is to implement a caching mechanism that stores the previously generated key and value vectors for reuse, which helps us to avoid unnecessary recomputations.
- Implementation brief walkthrough:
    - Registering the cache buffers with `self.register_buffer("cache_k", None)` and `self.register_buffer("cache_v", None)` inside the `MultiHeadAttention` constructor.
    - Forward pass with `use_cache` flag.
    - When generating texts, between independent sequences we must rest both buffers, so we also add a cache resetting method to the `MultiHeadAttention` class.
    - Propagating `use_cache` in the full model.
    - Using the cache in generation.
- **Good**: Computational efficiency increases.
- **Bad**: Memory usage increases linearly.
- Optimizing the KV cache implementation with pre-allocate memory and truncate cache via sliding window.

## Attention Alternatives
- **Grouped-Query Attention (GQA)** has become the new standard replacement for a more compute-efficient and parameter-efficient alternative to multi-head attention (MHA) in recent years. 
    - Unlike MHA, where each head has its own set of keys and values, to reduce memory usage, GQA groups multiple heads to share the same key and value projections. The heads have different queries, but share the same key and value for multiple query heads, which leads to lower memory usage and improved efficiency. It assumes that the number of key-value groups is chosen carefully, and it performs comparably to standard MHA as shown in **Llama 2**. In the extreme case where all attention heads share a single key-value group, known as multi-query attention, the memory usage decreases even more drastically but modeling performance can suffer. 
- **Multi-Head Latent Attention (MLA)**, used in **DeepSeek V2, V3**, and **R1**, offers a different memory-saving strategy that also pairs particularly well with KV caching. 
    - Instead of sharing key and value heads like GQA, MLA compresses the key and value tensors into a lower-dimensional space before storing them in the KV cache. At inference time, these compressed tensors are projected back to their original size before being used, which adds an extra matrix multiplication but reduces memory usage. The queries are also compressed, but only during training, not inference. It outperforms GQA in modeling performance.
- **Sliding Window Attention (SWA)** is applied in **Gemma 3**. If we think of regular self-attention as a **global** attention mechanism, since each sequence element can access every other sequence element, then SWA is like **local** attention, because here we restrict the context size around the current query position. Instead of attenting to all previous tokens, each token only attends to a fixed-size local window around its position, which lowers the size of the KV cache substantially. 
    - **Gemma 2** used a hybrid approach that combined local and global attention layers in a 1:1 ratio. **Gemma 3** then took the design further toward efficiency by using a 5:1 ratio between sliding window and full attention layers, which means for every 5 loca attention (SWA) layers, there is one global layer. In addition, the sliding window size was reduced from 4096 tokens in Gemma 2 to 1024 tokens in Gemma 3, but the technical report indicate that these changes have only a minor effect on overall model quality.
- **Gated DeltaNet for Linear Attention** was proposed by **Qwen3-Next** and **Kimi Linear**, used in hybrid transformers that implement alternatives to the attention mechanism that scale linearly instead of quadratically with respect to the context length. Both of them use a 3:1 ratio, meaning for every three transformer blocks employing the linear gated DeltaNet variant, there is one block that uses full attention. **Gated DeltaNet** is a linear attention variant with inspiration from recurrent neural networks. 
    - **Kimi Linear** modifies the linear attention mechanism of **Qwen3-Next** by the **Kimi Delta Attention (KDA)** mechanism, which is essentially a refinement of Gated DeltaNet. **Qwen3-Next** applies a scalar gate (one value per attention head) to control the memory decay gate, **Kimi Linear** replaces it with a channel-wise gating for each feature dimension, which gives more control over the memory, and in turn, imrpoves long-context reasoning. For full attention layers, **Kimi Linear** replaces Qwen3-Next's gated attention layers (which are essentially standard multi-head attention layers with output gating) with MLA (with an additional gate).
    - **Gated Attention**: After computing attention as usual, the model iuses a separate gating signal from the same input, applies a sigmoid to keep it between 0 and 1, and multiplies it with the attention output, which allows the model to scale up or down certain features dynamically, and it helps with training stability (eliminate issues like Attention Sink and Massive Attention).
    - **Gated DeltaNet** is **Qwen3-Next**'s linear-attention layer, which is intended as an alternative to standard softmax attention. The delta rule part refers to computing the difference (delta) between new and predicted values to update a hidden state that is used as a memory state. **Gated DeltaNet** has a gate similar to the gate in gated attention, except that it uses a SiLU instead of logistic sigmoid activation. 
        - A **Decay Gate** controls how fast the memory decays or resets over time.
        - An **Update Gate** controls how strongly new inputs modify the state.
    - In **Gated Attention**, the model computes normal attention between all tokens. Then, after getting the attention output, a gate (sigmoid) decides how much of that output to keep. The takeaway is that it is still the regular scaled-dot product attention that scales quadratically with context length. The Q and K are both n\*d matrices, where n is the number of input tokens, and d is the embedding dimension, and scaled-dot product attention results in an n\*n matrix multiplied by an n\*d dimensional value matrix V.
    - In **Gated DeltaNet**, there are no n\*n attention matrix. Instead, the model processes tokens one by one, and it keeps a running memory (a state) that gets updated as each new token comes in. 
    - So, in a sense, this state update in **Gated DeltaNet** is similar to how **RNN** work. The advantage is that it scales linearly instead of quadratically with context length, while the downside is that it sacrifices the global contex modeling ability that comes from full pairwise attention. 
    - **Gated DeltaNet** can still capture context to some extent, but it has to go through the memory bottleneck, which is a fixed size and thus more efficient, but it compresses past context into a single hidden state similar to **RNNs**, which is why **Qwen3-Next** and **Kimi Linear** don't replace all attention layers with DeltaNet layers but uses the 3:1 ratio.

## Mixture-of-Experts (MoE)
- The core idea in MoE is to replace each feed-forward module in a transformer block with multiple expert layers, where each of these expert layers is also a feed-forward module. This means we replace a single feed-forward block with multiple feed-forward blocks. 
- The feed-forward block inside a transformer block typlically contains a large  number of the model's total parameters. So, replacing a single feed-forward block with multiple feed-forward blocks substantially increases the model's total parameter count. However, the key trick is that we don't use (activate) all experts for every token, instead, a router selects only a small subset of experts per token.
- Because only a few experts are active at a time, MoE modules are often referred to as **sparse**, in contrast to **dense** modules that always use the full parameter set. However, the large total number of parameters via an MoE increases the capacity of the LLM, which means it can take up more knowledge during training. The sparsity keeps inference efficient, through, as we don't use all the parameters at the same time. 
- For example, **DeepSeek V3** has 256 experts per MoE module and a total of 671 billion parameters. Yet during inference, only 9 experts are active at a time (1 shared expert plus 8 selected by the router). This means just 37 billion parameters are used for each token inference step as opposed to all 671 billion. 
- The benefit of having a shared expert is that it boosts overall modeling performance compared to no shared experts. This is likely because common or repeated patterns don't have to be learned by multiple individual experts, which leaves them with more room for learning more specialized patterns. 

## Useful Links
- [BPE Algorithm](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)
- [Grouped-Query Attention (GQA)](https://arxiv.org/abs/2305.13245)
- [Llama 2 Paper](https://arxiv.org/abs/2307.09288)
- [DeepSeek V2 Paper](https://arxiv.org/abs/2405.04434)
- [Gemma 2 Paper](https://arxiv.org/abs/2408.00118)
- [Gemma 3 Paper](https://arxiv.org/abs/2503.19786)
- [LongFormer paper (introducing SWA)](https://arxiv.org/abs/2004.05150)
- [Qwen3-Next Paper](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)
- [Qwen3-Next Implementation](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835)
- [Kimi Linear Paper](https://arxiv.org/abs/2510.26692)
- [Gated Delta Networks in Mamba 2](https://arxiv.org/abs/2412.06464)
- [DeepSpeed-MoE 2022](https://arxiv.org/abs/2201.05596)
- [DeepSeekMOE 2024](https://arxiv.org/abs/2401.06066)