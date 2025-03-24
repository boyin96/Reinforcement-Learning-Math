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

   import copy
   import math
   
   import torch
   import torch.nn as nn
   import torch.nn.functional as F


   class LayerNorm(nn.Module):
   
       def __init__(self, features, eps=1e-6):
           super().__init__()
   
           self.a_2 = nn.Parameter(torch.ones(features))
           self.b_2 = nn.Parameter(torch.zeros(features))
           self.eps = eps
   
       def forward(self, x):
           mean = x.mean(-1, keepdim=True)
           std = x.std(-1, keepdim=True)
           return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
   
   
   def clones(module, n):
       return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


   def attention(query, key, value, mask=None, dropout=None):
       """Scaled Dot Product Attention."""
   
       d_k = query.size(-1)
       scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
       if mask is not None:
           scores = scores.masked_fill(mask == 0, -1e9)
       p_attn = scores.softmax(dim=-1)
       if dropout is not None:
           p_attn = dropout(p_attn)
       return torch.matmul(p_attn, value), p_attn


   class MultiHeadedAttention(nn.Module):
       def __init__(self, h, d_model, dropout=0.1):
   
           super().__init__()
   
           assert d_model % h == 0
   
           self.d_k = d_model // h
           self.h = h
           self.linears = clones(nn.Linear(d_model, d_model), 4)
           self.attn = None
           self.dropout = nn.Dropout(p=dropout)
   
       def forward(self, query, key, value, mask=None):
   
           if mask is not None:
               mask = mask.unsqueeze(1)
           b = query.size(0)
   
           query, key, value = [
               lin(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
               for lin, x in zip(self.linears, (query, key, value))
           ]
   
           x, self.attn = attention(
               query, key, value, mask=mask, dropout=self.dropout
           )
   
           x = (
               x.transpose(1, 2)
               .contiguous()
               .view(b, -1, self.h * self.d_k)
           )
           del query
           del key
           del value
           return self.linears[-1](x)


2. **PositionWiseFeedForward**

.. code-block:: python
   
   class PositionWiseFeedForward(nn.Module):
   
       def __init__(self, d_model, d_ff, dropout=0.1):
           super().__init__()
           self.w_1 = nn.Linear(d_model, d_ff)
           self.w_2 = nn.Linear(d_ff, d_model)
           self.dropout = nn.Dropout(dropout)
   
       def forward(self, x):
           return self.w_2(self.dropout(self.w_1(x).relu()))


3. **Positional Encoding**

Transformers do not have built-in sequential order awareness, unlike RNNs. Therefore, we need to inject position information explicitly. The positional encoding (PE) helps the model distinguish between different positions in a sequence by assigning unique vectors to each position.

The common approach is to use sinusoidal functions:

.. math::
   PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right),

.. math::
   PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right),

where:

- :math:`pos` is the position index in the sequence.
- :math:`i` is the dimension index.
- :math:`d_{\text{model}}` is the embedding size.

We analyze how :math:`PE(pos + k)` relates to :math:`PE(pos)`. Substituting :math:`pos + k` into the PE formula:

.. math::
   PE_{(pos+k, 2i)} = \sin\left(\frac{pos+k}{10000^{\frac{2i}{d_{\text{model}}}}}\right),

.. math::
   PE_{(pos+k, 2i+1)} = \cos\left(\frac{pos+k}{10000^{\frac{2i}{d_{\text{model}}}}}\right).

Using trigonometric sum identities:

.. math::
   \sin(A + B) = \sin A \cos B + \cos A \sin B,

.. math::
   \cos(A + B) = \cos A \cos B - \sin A \sin B.

Let :math:`\theta_i = \frac{1}{10000^{\frac{2i}{d_{\text{model}}}}}`, then:

.. math::
   PE_{(pos+k, 2i)} = \sin(pos\theta_i) \cos(k\theta_i) + \cos(pos\theta_i) \sin(k\theta_i),

.. math::
   PE_{(pos+k, 2i+1)} = \cos(pos\theta_i) \cos(k\theta_i) - \sin(pos\theta_i) \sin(k\theta_i).

This transformation can be rewritten as a **2D rotation matrix**:

.. math::
   \begin{bmatrix}
   PE_{(pos+k, 2i)} \\
   PE_{(pos+k, 2i+1)}
   \end{bmatrix} =
   \begin{bmatrix}
   \cos(k\theta_i) & \sin(k\theta_i) \\
   -\sin(k\theta_i) & \cos(k\theta_i)
   \end{bmatrix}
   \begin{bmatrix}
   PE_{(pos, 2i)} \\
   PE_{(pos, 2i+1)}
   \end{bmatrix}.

This means that moving from :math:`pos` to :math:`pos + k` is equivalent to rotating the positional encoding vector by an angle :math:`kθ_i`, where :math:`θ_i` depends on :math:`i`.

.. code-block:: python

   class PositionalEncoding(nn.Module):
   
       def __init__(self, d_model, dropout, max_len=5000):
           super().__init__()
           self.dropout = nn.Dropout(p=dropout)
   
           pe = torch.zeros(max_len, d_model)
           position = torch.arange(0, max_len).unsqueeze(1)
           div_term = torch.exp(
               torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
           )
           pe[:, 0::2] = torch.sin(position * div_term)
           pe[:, 1::2] = torch.cos(position * div_term)
           pe = pe.unsqueeze(0)
           self.register_buffer("pe", pe)
   
       def forward(self, x):
           x = x + self.pe[:, : x.size(1)].requires_grad_(False)
           return self.dropout(x)


4. **Encoder Structure**

.. code-block:: python


5. **Decoder Structure**

.. code-block:: python


6. **Encoder-Decoder Structure**

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


7. **Transformer**

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
