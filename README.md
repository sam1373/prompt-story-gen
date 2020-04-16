# Prompt-based Story Generation

Dataset: [https://github.com/pytorch/fairseq/tree/master/examples/stories](https://github.com/pytorch/fairseq/tree/master/examples/stories)

The model has been tested on the following requirements:
```
numpy==1.18.2
torch==1.3.0
transformers==2.8.0

Keras==2.3.1
tensorflow==1.15.0
```
Note: Tensorflow and Keras are only used to pad_sequences and load pretrained GPT2 weights. Main training happens on pyTorch.
