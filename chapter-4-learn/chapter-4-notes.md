# Chapter 4 Reading Notes

## Coding an LLM
- `qkv_bias` determines whether to include a bias vector in the `Linear` layers of the multi-head attention for query, key, and value computations. In the norm of modern LLMs, this is disabled, but it will be enabled when loading pretrained GPT-2 weights from OpenAI into our model
 
## Useful Links
- [Language Models Are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Lambda Labs](https://lambda.ai/)