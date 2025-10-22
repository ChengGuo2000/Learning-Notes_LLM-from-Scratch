# Chapter 7 Reading Notes

## Prepare the Instruction dataset
- Pre-trained LLMs can perform **text completion**, but they struggle with following specific instructions. **Instruction fine-tuning** involves training a model on a dataset where the input-output pairs are explicitly provided.
- There are two prompt styles for instruction fine-tuning in LLMs.
    - The **Alpaca** style uses a structured format with defined sections for `instruction`, `input`, and `response`. The input section may be left as blank.
    - The **Phi-3** style employs a simpler format with designated `<|user|>` and `<|assistant|>` tokens. The `<|user|>` part contains `instruction + ": \n" + input`, and the `<|assistant|>` part contains the `response`.
- A **collate** function, employed by the PyTorch `DataLoader` class, is responsible for taking a list of individual data samples and merging them into a single batch that can be processed efficiently by the model during training. 
- There are five steps involved in implementing the batching process, including applying the prompt template, using tokenization, adding padding tokens to ensure uniform length in each batch, creating target token IDs, and replacing `-100` placeholder tokens to mask padding tokens in the loss function.
- The PyTorch `cross_entropy` function ignores targets labeled with `-100` in its default setting, so we can take advantage of this to ignore the additional end-of-text padding tokens that we used to pad the training examples to have the same length in each batch. We will only keep one `<|endofteext|>` for LLM to learn that a response is complete.
- The `partial` function from the Python's `functools` standard library can create a new version of the function with some argument prefilled.

## Instruction Fine-tuning
- We are using the medium-sized GPT-2 because the smallest sized one is too limited in capacity to achieve satisfactory results via instruction fine-tuning. Smaller models lack the necessary capacity to learn and retain the intricate patterns and nuanced behaviors required for high-quality instruction-following tasks.
- Evaluating the instruction-fine-tuned LLMs such as chatbots has multiple approaches:
    - Short-answer and multiple-choice benchmarks, like the Measureing Massive Multitask Language Understanding (MMLU).
    - Human preference comparison to other LLMs, like LMSYS chatbot arena.
    - Automated conversational benchmarks, where another LLM is used to evaluate the responses, like AlpacaEval. 

## Useful Links
- [Instruction Tuning with Loss Over Instructions](https://arxiv.org/abs/2405.14394)
- [Stanford's Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca)
- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [LMSYS chatbot arena](https://lmarena.ai/)
- [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)