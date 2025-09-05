# Chapter 1 Reading Notes

## Building a LLM
- LLMs have broader proficiency than earlier NLP models.
- Utilizes next-word prediction.
- AI approaches that are not ML:
    - Rule-based Systems
    - Genetic Algorithms
    - Expert Systems
    - Fuzzy Logic
    - Symbolic Reasoning
- LLMs can be used for effective knowledge retrieval.
- Creating an LLM includes pretraining and fine-tuning.
    - Pretraining
        - Develop a broad understanding of language.
        - Large corpus of raw text (no labels).
        - Creating a base model for text completion and limited few-shot capabilities.
        - **Self-supervised learning**: model generates its own labels from the input data.
    - Fine-Tuning
        - Train on smaller labeled datasets for specific tasks.
        - Two varieties
            - **Classification fine-tuning**: the labeled dataset consists of texts and associated class labels.
            - **Instruction fine-tuning**: the labeled dataset consists of instruction and answer pairs.
        - LLMs fine-tuned on custom datasets can outperform general LLMs on specific tasks.

## Transformer
- Two Submodules
    - **Encoder**: From input text to vectors that capture the contextual information of the input.
    - **Decoder**: Takes encoded vectors and generates the output text one word at a time.
- Both the encoder and decoder are layers connected by a self-attention mechanism.
    - **Self-attention mechanism**: Allows the model to weigh the importance of different words or tokens in a sequence relative to each other.
        - Capture long-range dependencies and contextual relationships within the input data.
        - Enhancing the model's ability to generate coherent and contextually relevant output.
- **BERT**: Bidirectional Encoder Representations from Transformers
    - Built upon the original transformer's encoder submodule.
    - **Masked word prediction**: predicts masked or hidden words in a given sentence.
        - Strength in text classification
- **GPT**: Generative Pretrained Transformers
    - Focuses on the decoder submodule.
    - Designed for tasks that require generating texts.
    - For text completion tasks, and show versatility in capabilities.
    - Adept at zero-shot and few-shot learning tasks.
        - **Zero-shot learning**: the ability to generalize to completely unseen tasks without any prior specific examples.
        - **Few-shot learning**: learning from a minimal number of examples the user provides as input.
- We will pretrain an LLM, and also learn how to reuse only available model weights and load them into the architecture we will implement, so that we can skip the expensive pretraining stage.

## GPT
- **Next-word prediction**: self-supervised learning, which is a form of self-labeling.
    - We can use the next word as the label that the model is supposed to predict.
- Just a decoder part of the original transformer.
- **Autoregressive model**: incorporates their previous outputs as inputs for future predictions.
- Each new word is chosen based on the preceding sequence, which improves coherence.
- GPT-3 is trained on 300 billion tokens extracted from CommonCrawl, WebText2, Books1, Books2, and Wikipedia.
- GPT-3 has 96 transformer layers and 175 billion parameters.
- **Emergent behavior**: wasn't explicitly taught during training but emerges as a natural consequence of exposure to vast data in diverse contexts. GPT can also perform translation tasks.


## Useful Links
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762): Introducing Transformer.
- [Dolma: An Open Corpus of Three Trillion Tokens for LLM Pretraining Research](https://arxiv.org/pdf/2402.00159)
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): Introducing GPT
- [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155): OpenAI's InstructGPT paper.