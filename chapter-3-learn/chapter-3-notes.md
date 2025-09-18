# Chapter 3 Reading Notes

## Encoder-Decoder RNN
- **RNN**: a popular encoder-decoder architecture before transformers. RNN is a type of NN where outputs from previous steps are fed as inputs to the current step, making them well-suited for sequential data like text.
    - The input text is fed into the **encoder**, which processes it sequentially. 
    - The **encoder** updates its hidden state at each step, trying to capture the entire meaning of the input sentence in the final hidden state.
    - The **decoder** then takes this final hidden state to start generating the translated sentence, one word at a time.
    - The **decoder** also updates its hidden state at each step, which is supposed to carry the context necessary for the next-word prediction.
 - **Limitation of RNN**: RNN cannot directly access earlier hidden states from the encoder during the decoding phase, so it relies only on the current hidden state, which encapsulates all relevant information, which lead to a loss of context, especially in complex sentences where dependencies might span long distances.

## Simplified Self-Attention Mechanism
- **Bahdanau Attention Mechannism**: Since RNN must remember the entire encoded input in a single hidden state before passing it to the decoder, the researchers developed this mechanism which modifies the encoder-decoder RNN such that the decoder can selectively access different parts of the input sequence at each decoding step.
- Using an **attention mechanism**, the text-generating decoder part of the network can access all input tokens selectively, which means that some input tokens are more important than others for generating a given output token. The **importance** is determined by the **attention weights**.
- **Self-attention** is a mechanism that allows each position in the input sequence to consider the relevancy of, or **attend to**, all other positions in the same sequence when computing the representation of a sequence. It is a key component in modern LLMs like GPT.
- The **self** refers to the ability to compute attention weights by relating different positions within a single input sequence, in contrast to traditional attention mechanisms where the focus is on the relationships between elements of two different sequences.
- A **context vector** can be interpreted as an enriched embedding vector. The goal of self-attention is to compute a context vector for each input element in the input sequence that combines information from all other input elements.
- A **dot product** is essentially a concise way of multiplying two vectors element-wise and then summing the products. It is also a measure of similarity because it quantifies how closely two vectors are aligned. A higher dot product indicates a greater degree of alignment or similarity between the vectors. In self-attention mechanisms, the higher the dot product, the higher the similarity and attention score between two elements.
- The workflow for computing **context vectors** in the simplified version of self-attention:
    - Compute **attention scores**: the dot products between the inputs.
    - Conpute **attention weights**: a normalized version of the attention scores using softmax.
    - Compute **context vectors**: multiplying the embedded input tokens x^i with the corresponding attnetion weights and then summing the resulting vectors.
- The `dim = -1` in softmax applies the normalization along the last dimension of the tensor. For example, if `attention_scores` is a 2D tensor of [rows, colmns], then it will normalize across the columns so that the values in each row sum up to 1.

## Self-Attention Mechanism
- This mechanism is also called **scaled dot-product attention**. It introduced **weights matrices** that are updated during model training.
- There are three weight matrices.
    - **Query** ($W_q$): A query is analogous to a search query in a database. It represents the current item the model focuses on or tries to understand. The query is used to probe the other parts of the input sequence to determine how much attention to pay to them.
    - **Key** ($W_k$): The key is like a database key used for indexing and searching. Each item in the input sequence has an associated key. These keys are used to match the query.
    - **Value** ($W_v$): The value is similar to the value in a key-value pair in a database. It represents the actual content or representation fo the input items. Once the model determines which keys (which parts of the input) are the most relevant to the query (the current focus item), it retrieves the corresponding values.
- **Attention weights** determine the extent to which the network focuses on different parts of the input. They are dynamic, context-specific values.
- **Weight parameters** are the fundamental, learned coefficients that define the network's connections. They are optimized during training.
- The **attention score** computation is a dot-product computation using the query and key obtained by transforming the inputs via the respective weight matrices.
- When computing **attention weights**, we scale the **attention scores** by dividing them by the square root of the embedding dimension of the keys, then apply softmax.
- The reason for the **normalization by the embedding dimension size** is to improve the training performance by avoiding small gradients. As the dot products increase, the softmax function behaves more like a step function, resulting in gradients nearing zero, which can drastically slow down learning or cause training to stagnate. That is also why this is mechanism is called **scaled dot-product attention**.
- The final step of the **self-attention computation** is to compute the **context vector** by combining all value vectors via the attention weights.

## Useful Links
- [Attention is All You Need (2017)](https://arxiv.org/pdf/1706.03762)
- [Bahdanau Attention Mechansim for RNNs (2014)](https://arxiv.org/pdf/1409.0473)