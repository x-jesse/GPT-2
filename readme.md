# OpenAI GPT-2 Re-Implementation

This is a re-implementation of the GPT-2 (124M) model in PyTorch based on the OpenAI paper [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and Andrej Karpathy's Youtube series [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

### Usage

To run the model:
```
```


## Table of Contents

1. [Introduction](#introduction)
    - [What is a GPT?](#what-is-a-gpt)
    - [Transformers - "Attention is All You Need"]()
2. [Implementation]()
    - [Building the Model]()
    - [Training the Model]()

## Introduction


### What is a GPT?
Formally, a generative pre-trained transformer is a LLM that uses transformer-type architecture to perform predictive text generation: As introduced in the 2017 paper [Attention is All You Need](https://arxiv.org/abs/1706.03762), transfomers use the principle of "attention" to provide contextual weight to token embeddings. 

In other words, a GPT is just a really big game of guess-the-next-word.