Understanding GPT as an Attention-Driven Decoder
================================================

Introduction
------------
The Generative Pre-trained Transformer (GPT) is a state-of-the-art autoregressive language model that uses the Transformer architecture with a decoder-only design. Unlike traditional sequence-to-sequence models with both encoder and decoder, GPT relies solely on a stack of masked self-attention layers to generate coherent and contextually relevant text.

GPT is widely applied in natural language processing (NLP) tasks such as text generation, summarization, and dialogue systems. It builds on the fundamental principles of the "Attention is All You Need" paper by using self-attention mechanisms to capture long-range dependencies in text.

Principles
----------
GPT operates as an autoregressive model, meaning it generates text token by token, conditioning each token’s prediction on the previous tokens. The key components of GPT’s architecture are:

1. **Token Embeddings**: Input text is tokenized and mapped to dense vector representations.
2. **Positional Encodings**: Since self-attention does not inherently capture token order, GPT uses learned positional embeddings.
3. **Masked Multi-Head Self-Attention**: Each token attends only to previous tokens, ensuring unidirectional information flow.
4. **Feed-Forward Networks (FFN)**: Each attention layer is followed by a position-wise feed-forward network.
5. **Layer Normalization and Residual Connections**: Applied to stabilize training and improve gradient flow.
6. **Softmax Output Layer**: The final output probabilities are computed using a softmax function over the vocabulary.

Mathematical Formulation
------------------------
The core of GPT’s text generation relies on the masked self-attention mechanism. Given a sequence of tokens, each token’s representation is computed as follows:

.. math::
   
   Q = XW_Q, \quad K = XW_K, \quad V = XW_V

where:

- :math:`X \in \mathbb{R}^{T \times d_{model}}` is the input token representation.
- :math:`W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}` are learned weight matrices for queries, keys, and values.

The attention scores are then computed using the scaled dot-product attention:

.. math::
   
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V

where :math:`M` is a lower triangular mask matrix with negative infinity in masked positions, ensuring that each token attends only to previous tokens.

Code Implementation
-------------------
The following PyTorch code demonstrates the simplified GPT-like layers:

1. **Scaled DotProduct Attention Layer**

.. code-block:: python

   import torch
   import torch.nn as nn


   class ScaledDotProductAttention(nn.Module):
       def __init__(self, d_k, attn_dropout=0.1):
           super().__init__()
           
           self.d_k = d_k

           self.dropout = nn.Dropout(attn_dropout)
           self.softmax = nn.Softmax(dim=-1)
       
       def forward(self, q, k, v, mask=None):
           
           # q -> (batch_size, n_heads, q_len, d_k)
           # k -> (batch_size, n_heads, k_len, d_k)
           # v -> (batch_size, n_heads, v_len, d_v)
           # mask -> (batch_size, n_heads, q_len, k_len)
           
           attn_score = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

           if mask is not None:
               attn_score.masked_fill_(mask, -1e9)  # attn_score -> (batch_size, n_heads, q_len, k_len)
           
           attn_weights = self.dropout(self.softmax(attn_score))  # attn_weights -> (batch_size, n_heads, q_len, k_len)
           output = torch.matmul(attn_weights, v)  # output -> (batch_size, n_heads, q_len, d_v)
   
           return output, attn_weights

2. **MultiHead Attention Layer**

.. code-block:: python

   class MultiHeadAttention(nn.Module):
       def __init__(self, d_model, n_heads, attn_dropout):
           super().__init__()

           self.n_heads = n_heads
           self.d_k = self.d_v = d_model // n_heads
   
           self.WQ = nn.Linear(d_model, d_model)
           self.WK = nn.Linear(d_model, d_model)
           self.WV = nn.Linear(d_model, d_model)
   
           self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k, attn_dropout)
   
           self.fc = nn.Linear(d_model, d_model)

       def forward(self, q, k, v, mask=None):

           # q -> (batch_size, q_len(=seq_len), d_model)
           # k -> (batch_size, k_len(=seq_len), d_model)
           # v -> (batch_size, v_len(=seq_len), d_model)
           # mask -> (batch_size, q_len, k_len)
   
           batch_size = q.size(0)
   
           # q_heads -> (batch_size, n_heads, q_len, d_k)
           # k_heads -> (batch_size, n_heads, k_len, d_k)
           # v_heads -> (batch_size, n_heads, v_len, d_v)
           q_heads = self.WQ(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
           k_heads = self.WK(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
           v_heads = self.WV(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
   
           # mask -> (batch_size, n_heads, q_len, k_len)
           # attn -> (batch_size, n_heads, q_len, d_v)
           # attn_weights -> (batch_size, n_heads, q_len, k_len)
           if mask is not None:
               mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
           attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, mask=mask)
   
           # attn -> (batch_size, q_len, n_heads * d_v)
           # outputs -> (batch_size, q_len, d_model)
           attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
           outputs = self.fc(attn)
   
           return outputs, attn_weights

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class MaskedSelfAttention(nn.Module):
       def __init__(self, embed_dim, num_heads):
           super().__init__()
           self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
       
       def forward(self, x):
           seq_length = x.size(1)
           mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)  # Lower triangular mask
           mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
           
           attn_output, _ = self.attention(x, x, x, attn_mask=mask)
           return attn_output

This module ensures that each token can only attend to previous tokens, enforcing the autoregressive property of GPT.

Conclusion
----------
GPT’s decoder-only architecture, powered by masked self-attention, enables it to generate high-quality text by leveraging contextual information effectively. Its autoregressive nature ensures that text is generated in a coherent and grammatically accurate manner. The use of multi-head self-attention allows for capturing complex dependencies, making GPT a powerful model for various NLP tasks.

References
--------------------
- `Attention Is All You Need <https://arxiv.org/pdf/1706.03762>`_
- `Improving Language Understanding by Generative Pre-Training <https://www.mikecaptain.com/resources/pdf/GPT-1.pdf>`_
