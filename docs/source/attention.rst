Attention Is All You Need
======================================

Introduction
-------------
The Transformer model, introduced in the paper "Attention Is All You Need" by Vaswani et al., revolutionized natural language processing (NLP) by replacing recurrent and convolutional layers with a self-attention mechanism. This architecture enables highly parallel computation and captures long-range dependencies efficiently.

Transformers are widely used in machine translation, text summarization, and other NLP tasks. Their encoder-decoder structure makes them versatile for both sequence-to-sequence and autoregressive tasks.

Architecture Overview
----------------------
The Transformer consists of two main components:

1. **Encoder**: Processes the input sequence and generates contextualized representations.
2. **Decoder**: Generates the output sequence, conditioned on the encoder’s representations and previously generated tokens.

Each encoder and decoder block contains:

- **Multi-Head Self-Attention**: Allows each token to attend to all tokens in the sequence.
- **Feed-Forward Network (FFN)**: Applies a non-linear transformation to each token’s representation.
- **Layer Normalization and Residual Connections**: Stabilizes training and facilitates gradient flow.
- **Positional Encoding**: Injects sequential information since self-attention lacks inherent order awareness.

Self-Attention Mechanism
-------------------------
The core of the Transformer is the scaled dot-product attention, which computes the attention scores as follows:

.. math::
   
   Q = XW_Q, \quad K = XW_K, \quad V = XW_V,

where :math:`X \in \mathbb{R}^{T \times d_{model}}` is the input token representation, and :math:`W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}` are learned weight matrices.

The attention weights are computed using:

.. math::
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.

This mechanism allows each token to selectively focus on other tokens in the sequence.

Multi-Head Attention
---------------------
Instead of a single attention function, Transformers use multiple attention heads:

.. math::
   \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O,

where each head independently performs self-attention. This enhances the model’s ability to capture different aspects of dependencies in the data.

Code Implementation
--------------------
Below is a PyTorch implementation:

.. code-block:: python

   import torch
   import torch.nn as nn

   class SelfAttention(nn.Module):
       def __init__(self, embed_dim, num_heads):
           super().__init__()
           self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
       
       def forward(self, x):
           attn_output, _ = self.attention(x, x, x)
           return attn_output

Conclusion
-----------
The Transformer model has become the foundation of modern NLP due to its efficient self-attention mechanism and parallel computation capabilities. By eliminating recurrence, it enables faster training and better captures long-range dependencies. Understanding its architecture is crucial for leveraging state-of-the-art language models like BERT and GPT.

References
------------
- `Attention Is All You Need <https://arxiv.org/pdf/1706.03762>`_
- https://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://jalammar.github.io/illustrated-transformer/
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
