# Chapter 6 Reading Notes

## Fine-tuning
- **Instruction fune-tuning** involves training a language model on a set of tasks using specific instructions to improve its ability to understand and execute tasks described in natural language prompts. It typically can undertake a broader range of tasks.
- **Classification fine-tuning** is hwen the model is trained to recognize a specific set of class labels. It is restricted to predicting classes it has encountered during its training, so it is highly specialized.
- **Instruction fine-tuning** improves the model's ability to understand and generate responses based on specific user instructions. It is best suited for models that need to handle a variety of tasks. It demands larger datasets and greater computational resources to develop models proficient in various tasks.
- **Classification fine-tuning** is ideal for projects requiring precise categorization of data into predefined classes. It requires less data and compute power, but it is confined to the specific classes it trained on.
- For **spam detection**, since we are working with a dataset that contains texts of varying lengths, we are going to pad all messages to the length of the longest message in the dataset or batch, because if we truncate, it may reduce model performance. We can use `<|endoftext|>` as a padding token. We may truncate text for validation and test sets.

## Useful Links