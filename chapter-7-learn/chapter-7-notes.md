# Chapter 7 Reading Notes

## Instruction Fine-tuning
- Pre-trained LLMs can perform **text completion**, but they struggle with following specific instructions. **Instruction fine-tuning** involves training a model on a dataset where the input-output pairs are explicitly provided.
- There are two prompt styles for instruction fine-tuning in LLMs.
    - The **Alpaca** style uses a structured format with defined sections for `instruction`, `input`, and `response`. The input section may be left as blank.
    - The **Phi-3** style employs a simpler format with designated `<|user|>` and `<|assistant|>` tokens. The `<|user|>` part contains `instruction + ": \n" + input`, and the `<|assistant|>` part contains the `response`.

## Useful Links
