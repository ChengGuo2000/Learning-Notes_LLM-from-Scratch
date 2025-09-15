# Chapter 2 Reading Notes

## Word Embeddings
- **Embedding**: converting data into a vector format, where different data format requires different embedding models, mapping discrete objects to points in a continuous vector space, and converting nonnumeric data into a NN-processable format.
- **Retrieval-Augmented Generation (RAG)** uses sentence or paragraph embeddings, and combines generation with retrieval to pull relevant information when generating text.
- **Word2Vec** is a pretrained model, which trains NN artchitecture to generate word embeddings by predicting the context of a word given the target word or vice versa. The main idea is that words appear in similar contexts tend to have similar meanings. When projected into 2D space for visualization, similar terms are clustered together.
- For GPT-2 and GPT-3, the **embedding size** varies based on the specific model variant and size. The embedding size is often referred to as the dimensionality of the model's hidden states.
    - The smallest GPT-2 models (117M and 125M parameters) use an embedding size of 768 dimensions.
    - The largest GPT-3 model (175B parameters) uses an embedding size of 12288 dimensions.
- **Whitespace** in tokenizer
    - If **remove**, reduces the memory and computing requirements.
    - If **keep**, can be useful if we train models that are sensitive to the exact structure of the text (like Python code, which is sensitive to indentation and spacing).

## Special Tokens
- `<|unk|>`: for words that are not part of the vocabulary.
- `<|endoftext|>`: separate two unrelated text sources.
- `[BOS]`: beginning of sequence.
- `[EOS]`: end of sequence, similar to `<|endoftext|>`.
- `[PAD]`: padding, when training LLMs with batch size larger than one, the batch might contain texts of varying lengths. To ensure all texts have the same length, the shorter texts are extended or "padded" using this token, up to the length of the longest text in the batch.
- GPT only uses `<|endoftext|>` tokens for simplicity, which is analogous to `[EOS]` and can also be used for padding. It also doesn't use `<|unk|>` tokens, instead using **byte pair encoding** tokenizer.
- We will use mask, meaning we don't addtend to padded tokens, so the specific token chosen for padding is inconsequential.

## Byte Pair Encoding (BPE)
- The BPE tokenizer used in training GPT-2 and GPT-3 has a total vocab size of 50257. 
- The BPE algorithm breaks down words that aren't in its predefined vocabulary into smaller subword units or even individual characters, enabling it to handle out-of-vocab words. This ability of break down allows trained LLM to process any texts.
- BPE builds its vocabulary by iteratively merging frequent characters into subwords, then into words.
- It starts with adding all individual single characters into vocab, then it merge character combinations that frequently occur together into subwords, like merging `d` and `e` into `de`.

## Data Sampling, Token Embeddings, and Positional Embeddings
- Most LLMs train with input sizes (`max_length`) of at least 256.
- The `stride` setting dictates the number of positions the input shift across batches, emulating a sliding window approach.
- `batch_size` is a tradeoff and a hyperparameter to experiment with when training LLMs. Small batch sizes require less memory during training but lead to more noisy model updates.
- The **weight matrix** of the embedding layer contains small, random values, which are optimized during LLM training. 
- LLM's **self-attention mechanism** doesn't have a notion of position or order for the tokens within a sequence.
- **Absolute positional embeddings**: directly associated with specific positions in a sequence. For each position in the input sequence, a unique embedding is added to the token's embedding to convey its exact location.
- **Relative positional embeddings**: on the relative position or distance between tokens. The model learns the relationships in terms of "how far apart", so that it can generalize better to sequence of varying lengths.
- GPT models use absolute positional embeddings that are optimized during the training process rather than being fixed or predefined like the positional encodings in the original transformer model.

## Useful Links
- ["The Pit and the Pendulum" by Edgar Allan Poe on Wikisource](https://en.wikisource.org/wiki/The_Works_of_the_Late_Edgar_Allan_Poe_(1850)/Volume_1/The_Pit_and_the_Pendulum): I chose this short story for text preprocessing to get some different results from using "The Verdict" by Edith Wharton.
- [Tiktoken - Python open source library for BPE](https://github.com/openai/tiktoken)
- [One-hot encoding followed by matrix multiplication in a fully connected layer](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/03_bonus_embedding-vs-matmul)