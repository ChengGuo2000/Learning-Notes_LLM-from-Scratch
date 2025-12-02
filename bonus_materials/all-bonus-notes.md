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


## Useful Links
- [BPE Algorithm](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)