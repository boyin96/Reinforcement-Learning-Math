Understanding GPT as an Attention-Driven Decoder
================================================

Introduction
------------
The Generative Pre-trained Transformer (GPT) is a state-of-the-art autoregressive language model that uses the Transformer architecture with a decoder-only design. Unlike traditional sequence-to-sequence models with both encoder and decoder, GPT relies solely on a stack of masked self-attention layers to generate coherent and contextually relevant text.

GPT is widely applied in natural language processing (NLP) tasks such as text generation, summarization, and dialogue systems. It builds on the fundamental principles of the "Attention is All You Need" paper by using self-attention mechanisms to capture long-range dependencies in text.

Decoder-Only Architecture
------------------------------------------

GPT adopts a decoder-only structure primarily because it is designed for autoregressive text generation. Unlike models with both an encoder and a decoder, GPT does not require an input sequence to be fully processed before generating output. Instead, it predicts each token sequentially, conditioning on the previously generated tokens.

Key reasons for using a decoder-only architecture include:

1. **Autoregressive Nature**: GPT generates text one token at a time, making it suitable for tasks like text completion, dialogue generation, and creative writing.
2. **Masked Self-Attention**: By applying a causal mask, GPT ensures that each token attends only to previous tokens, preventing information leakage from future tokens.
3. **Simplified Training Process**: The absence of an encoder simplifies training, as the model learns to predict the next token given a sequence of preceding tokens.
4. **Unidirectional Context**: Unlike bidirectional models (e.g., BERT), which consider both past and future context, GPT relies solely on past tokens, making it effective for generative tasks.

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
   
   Q = XW_Q, \quad K = XW_K, \quad V = XW_V,

where :math:`X \in \mathbb{R}^{T \times d_{model}}` is the input token representation and :math:`W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}` are learned weight matrices for queries, keys, and values.

The attention scores are then computed using the scaled dot-product attention:

.. math::
   
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V,

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

3. **Position-wise Feed-Forward Layer**

.. code-block:: python

   class PositionWiseFeedForwardNetwork(nn.Module):
       def __init__(self, d_model, d_ff):
           super().__init__()
   
           self.linear1 = nn.Linear(d_model, d_ff)
           self.linear2 = nn.Linear(d_ff, d_model)
           self.gelu = nn.GELU()
   
           nn.init.normal_(self.linear1.weight, std=0.02)
           nn.init.normal_(self.linear2.weight, std=0.02)
   
       def forward(self, inputs):
   
           # inputs -> (batch_size, seq_len, d_model)
   
           outputs = self.gelu(self.linear1(inputs))  # outputs -> (batch_size, seq_len, d_ff)
           outputs = self.linear2(outputs)  # outputs -> (batch_size, seq_len, d_model)
   
           return outputs

4. **Decoder Layer**

.. code-block:: python

   class DecoderLayer(nn.Module):
       def __init__(self, d_model, n_heads, d_ff, attn_dropout, resid_dropout):
           super().__init__()
   
           self.mha = MultiHeadAttention(d_model, n_heads, attn_dropout)
           self.dropout1 = nn.Dropout(resid_dropout)
           self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-5)
   
           self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
           self.dropout2 = nn.Dropout(resid_dropout)
           self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-5)
   
       def forward(self, inputs, mask=None):
   
           # inputs -> (batch_size, seq_len, d_model)
           # mask -> (batch_size, seq_len, seq_len)
   
           attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, mask=mask)
   
           # attn_outputs -> (batch_size, seq_len, d_model)
           # attn_weights -> (batch_size, n_heads, q_len(=seq_len), k_len(=seq_len))
           attn_outputs = self.dropout1(attn_outputs)
           attn_outputs = self.layer_norm1(inputs + attn_outputs)
   
           ffn_outputs = self.ffn(attn_outputs)
           ffn_outputs = self.dropout2(ffn_outputs)
           ffn_outputs = self.layer_norm2(attn_outputs + ffn_outputs)  # ffn_outputs -> (batch_size, seq_len, d_model)
   
           return ffn_outputs, attn_weights

5. **Transformer Decoder**

.. code-block:: python

   class TransformerDecoder(nn.Module):
       def __init__(self, vocab_size, seq_len, d_model, n_layers, n_heads, d_ff,
                    embd_dropout, attn_dropout, resid_dropout, pad_id):
           super().__init__()
   
           self.pad_id = pad_id
   
           # layers
           self.embedding = nn.Embedding(vocab_size, d_model)
           self.dropout = nn.Dropout(embd_dropout)
           self.pos_embedding = nn.Embedding(seq_len + 1, d_model)
           self.layers = nn.ModuleList(
               [DecoderLayer(d_model, n_heads, d_ff, attn_dropout, resid_dropout) for _ in range(n_layers)]
           )
   
           nn.init.normal_(self.embedding.weight, std=0.02)
   
       def forward(self, inputs):
   
           # inputs -> (batch_size, seq_len)
           positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
           position_pad_mask = inputs.eq(self.pad_id)
           positions.masked_fill_(position_pad_mask, 0)  # positions -> (batch_size, seq_len)
   
           # outputs -> (batch_size, seq_len, d_model)
           outputs = self.dropout(self.embedding(inputs)) + self.pos_embedding(positions)
   
           # attn_pad_mask -> (batch_size, seq_len, seq_len)
           attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
   
           # subsequent_mask -> (batch_size, seq_len, seq_len)
           subsequent_mask = self.get_attention_subsequent_mask(inputs).to(device=attn_pad_mask.device)
   
           # attn_mask -> (batch_size, seq_len, seq_len)
           attn_mask = torch.gt((attn_pad_mask.to(dtype=subsequent_mask.dtype) + subsequent_mask), 0)
   
           attention_weights = []
           for layer in self.layers:
   
               # outputs -> (batch_size, seq_len, d_model)
               # attn_weights -> (batch_size, n_heads, seq_len, seq_len)
               outputs, attn_weights = layer(outputs, attn_mask)
               attention_weights.append(attn_weights)
   
           return outputs, attention_weights
   
       @staticmethod
       def get_attention_padding_mask(q, k, pad_id):
   
           # attn_pad_mask -> (batch_size, q_len, k_len)
           attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
   
           return attn_pad_mask
   
       @staticmethod
       def get_attention_subsequent_mask(q):
   
           bs, q_len = q.size()
           subsequent_mask = torch.ones(bs, q_len, q_len).triu(diagonal=1)  # subsequent_mask -> (batch_size, q_len, q_len)
   
           return subsequent_mask

5. **GPT**

.. code-block:: python

   class GPT(nn.Module):
       def __init__(
           self,
           vocab_size,
           seq_len=512,
           d_model=768,
           n_layers=12,
           n_heads=12,
           d_ff=3072,
           embd_dropout=0.1,
           attn_dropout=0.1,
           resid_dropout=0.1,
           pad_id=0,
       ):
           super().__init__()
   
           self.decoder = TransformerDecoder(vocab_size, seq_len, d_model, n_layers, n_heads, d_ff,
                                             embd_dropout, attn_dropout, resid_dropout, pad_id)
   
       def forward(self, inputs):
   
           # inputs -> (batch_size, seq_len)
   
           # outputs -> (batch_size, seq_len, d_model)
           # attention_weights -> [(batch_size, n_heads, seq_len, seq_len)] * n_layers
           outputs, attention_weights = self.decoder(inputs)
   
           return outputs, attention_weights
   
   
   class GPTLMHead(nn.Module):
       def __init__(self, gpt):
           super().__init__()
   
           vocab_size, d_model = gpt.decoder.embedding.weight.size()
   
           self.gpt = gpt
           self.linear = nn.Linear(d_model, vocab_size, bias=False)
           self.linear.weight = gpt.decoder.embedding.weight
   
       def forward(self, inputs):
   
           # inputs -> (batch_size, seq_len)
   
           # outputs -> (batch_size, seq_len, d_model)
           # attention_weights -> [(batch_size, n_heads, seq_len, seq_len)] * n_layers
           outputs, attention_weights = self.gpt(inputs)
   
           # lm_logits -> (batch_size, seq_len, vocab_size)
           lm_logits = self.linear(outputs)
   
           return lm_logits
   
   
   class GPTClsHead(nn.Module):
       def __init__(self, gpt, n_class, cls_token_id, cls_dropout=0.1):
           super().__init__()
   
           vocab_size, d_model = gpt.decoder.embedding.weight.size()
           self.cls_token_id = cls_token_id
   
           self.gpt = gpt
   
           # LM
           self.linear1 = nn.Linear(d_model, vocab_size, bias=False)
           self.linear1.weight = gpt.decoder.embedding.weight
   
           # Classification
           self.linear2 = nn.Linear(d_model, n_class)
           self.dropout = nn.Dropout(cls_dropout)
   
           nn.init.normal_(self.linear2.weight, std=0.02)
           nn.init.normal_(self.linear2.bias, 0)
   
       def forward(self, inputs):
   
           # inputs -> (batch_size, seq_len)
   
           # outputs -> (batch_size, seq_len, d_model)
           # attention_weights -> [(batch_size, n_heads, seq_len, seq_len)] * n_layers
           outputs, attention_weights = self.gpt(inputs)
   
           # lm_logits -> (batch_size, seq_len, vocab_size)
           lm_logits = self.linear1(outputs)
   
           # outputs -> (batch_size, d_model)
           # cls_logits -> (batch_size, n_class)
           outputs = outputs[inputs.eq(self.cls_token_id)]
           cls_logits = self.linear2(self.dropout(outputs))
   
           return lm_logits, cls_logits


Conclusion
----------
GPT’s decoder-only architecture, powered by masked self-attention, enables it to generate high-quality text by leveraging contextual information effectively. Its autoregressive nature ensures that text is generated in a coherent and grammatically accurate manner. The use of multi-head self-attention allows for capturing complex dependencies, making GPT a powerful model for various NLP tasks.

References
--------------------
- `Attention Is All You Need <https://arxiv.org/pdf/1706.03762>`_
- `Improving Language Understanding by Generative Pre-Training <https://www.mikecaptain.com/resources/pdf/GPT-1.pdf>`_
- https://github.com/lyeoni/gpt-pytorch
