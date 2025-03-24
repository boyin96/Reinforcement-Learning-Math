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

1. **Attention**

.. code-block:: python



1. **Encoder-decoder Structure**

.. code-block:: python

   import copy
   
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   
   
   class EncoderDecoder(nn.Module):
       """A standard Encoder-Decoder architecture. """
   
       def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
           super().__init__()
   
           self.encoder = encoder
           self.decoder = decoder
           self.src_embed = src_embed
           self.tgt_embed = tgt_embed
           self.generator = generator
   
       def forward(self, src, tgt, src_mask, tgt_mask):
           return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
   
       def encode(self, src, src_mask):
           return self.encoder(self.src_embed(src), src_mask)
   
       def decode(self, memory, src_mask, tgt, tgt_mask):
           return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
   
   
   class Generator(nn.Module):
   
       def __init__(self, d_model, vocab):
           super().__init__()
   
           self.proj = nn.Linear(d_model, vocab)
   
       def forward(self, x):
           return F.log_softmax(self.proj(x), dim=-1)

2. **Encoder**

.. code-block:: python



3. **Transformer**

.. code-block:: python

   def make_model(src_vocab, tgt_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
   
       c = copy.deepcopy
       attn = MultiHeadedAttention(h, d_model)
       ff = PositionwiseFeedForward(d_model, d_ff, dropout)
       position = PositionalEncoding(d_model, dropout)
       model = EncoderDecoder(
           Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n),
           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
           nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
           nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
           Generator(d_model, tgt_vocab),
       )
   
       for p in model.parameters():
           if p.dim() > 1:
               nn.init.xavier_uniform_(p)
       return model

Conclusion
-----------
The Transformer model has become the foundation of modern NLP due to its efficient self-attention mechanism and parallel computation capabilities. By eliminating recurrence, it enables faster training and better captures long-range dependencies. Understanding its architecture is crucial for leveraging state-of-the-art language models like BERT and GPT.

References
------------
- `Attention Is All You Need <https://arxiv.org/pdf/1706.03762>`_
- https://nlp.seas.harvard.edu/annotated-transformer/
- https://jalammar.github.io/illustrated-transformer/
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
