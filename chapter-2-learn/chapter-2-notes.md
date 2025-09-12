# Chapter 2 Reading Notes

## Word Embeddings
- **Embedding**: converting data into a vector format, where different data format requires different embedding models, mapping discrete objects to points in a continuous vector space, and converting nonnumeric data into a NN-processable format.
- **Retrieval-Augmented Generation (RAG)** uses sentence or paragraph embeddings, and combines generation with retrieval to pull relevant information when generating text.
- **Word2Vec** is a pretrained model, which trains NN artchitecture to generate word embeddings by predicting the context of a word given the target word or vice versa. The main idea is that words appear in similar contexts tend to have similar meanings. When projected into 2D space for visualization, similar terms are clustered together.
- For GPT-2 and GPT-3, the **embedding size** varies based on the specific model variant and size. The embedding size is often referred to as the dimensionality of the model's hidden states.
    - The smallest GPT-2 models (117M and 125M parameters) use an embedding size of 768 dimensions.
    - The largest GPT-3 model (175B parameters) uses an embedding size of 12288 dimensions.
- Whitespace in tokenizer
    - If **remove**, reduces the memory and computing requirements.
    - If **keep**, can be useful if we train models that are sensitive to the exact structure of the text (like Python code, which is sensitive to indentation and spacing).

## Useful Links
- ["The Pit and the Pendulum" by Edgar Allan Poe on Wikisource](https://en.wikisource.org/wiki/The_Works_of_the_Late_Edgar_Allan_Poe_(1850)/Volume_1/The_Pit_and_the_Pendulum): I chose this short story for text preprocessing to get some different results from using "The Verdict" by Edith Wharton.