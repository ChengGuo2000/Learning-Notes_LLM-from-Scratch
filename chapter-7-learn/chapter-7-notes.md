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
- To further improve our model's performance, we can explore the following strategies:
    - Adjusting the hyperparameters during fine-tuning (learning rate, batch size, number of epochs).
    - Increasing the size of the training dataset.
    - Diversifying the examples in the trainign dataset to cover a broader range of topics and styles.
    - Experimenting with different prompts or instruction formats.
    - Using a larger pretrained model. 

## Using Ollama for Evaluation
- Start Ollama with `ollama serve`.
- Try the Llama 3 model with `ollama run llama3`.
- End the current Ollama session with `/bye`.

## Conclusion
- We finished implementing an LLM architecture, pretraining an LLM, and fine-tuning it for specific tasks. 
- One optional step after instruction fine-tuning is **preference fine-tuning**, which is particularly useful for customizing a model to better align with specific user preferences. 
- The field of AI and LLM research is evolving rapidly, here are some ways to keep up with the latest advancements:
    - Explore recent research papers on **arXiv**.
    - Check the author's blog.
    - Many researchers and practitioners are very active in sharing and discussing the latest developments on social media on X and Reddit, like the subreddit `r/LocalLLaMA`, which is a good resource for connecting with the community and staying informed.
- We can also utilize different and more powerful LLMs for real-world applications, including **Axolotl** and **LitGPT**.
- Building an LLM from scratch is the most effective way to gain a deep understanding of how LLMs work.
- Thank you author for this wonderful journey. 

## Useful Links
- [Instruction Tuning with Loss Over Instructions](https://arxiv.org/abs/2405.14394)
- [Stanford's Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca)
- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [LMSYS chatbot arena](https://lmarena.ai/)
- [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)
- [Preference tuning with DPO](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/04_preference-tuning-with-dpo)