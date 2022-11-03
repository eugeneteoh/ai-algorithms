import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        A = Q @ K.transpose(2, 3)
        A /= np.sqrt(self.d_k)
        A = F.softmax(A, dim=-1)
        return A @ V

    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.shape

        Q = self.W_q(X_q)
        K = self.W_k(X_k)
        V = self.W_v(X_v)

        # Split heads
        Q = Q.view(batch_size, self.num_heads, seq_length, self.d_k)
        K = K.view(batch_size, self.num_heads, seq_length, self.d_k)
        V = V.view(batch_size, self.num_heads, seq_length, self.d_v)

        H_cat = self.scaled_dot_product_attention(Q, K, V)
        H_cat = H_cat.view(batch_size, seq_length, dim)

        return self.W_o(H_cat)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, ff_hidden_dim=2048):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model),
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):
        out = self.layernorm1(x + self.mha(x, x, x))
        out = self.layernorm2(out + self.ff(out))
        return out


class Embedding(nn.Module):
    def __init__(self, d_model=512, vocab_size=10000, max_len=5000):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.register_buffer(
            "positional_encodings", self.get_positional_encoding(d_model, max_len)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def get_positional_encoding(self, d_model, max_len):
        encodings = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2)
        denominator = 10000 ** (two_i / d_model)
        div = position / denominator
        encodings[:, 0::2] = torch.sin(div)
        encodings[:, 1::2] = torch.cos(div)
        encodings.requires_grad_(False)

        return encodings

    def forward(self, x):
        seq_length = x.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=x.device
        )  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(x)  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(x)

        embeddings = word_embeddings + self.positional_encodings

        return self.layernorm(embeddings)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        max_seq_len=5000,
        num_layers=6,
        d_model=512,
        num_heads=8,
        ff_hidden_dim=2048,
    ):
        super().__init__()

        self.embedding_layer = Embedding(
            d_model=d_model, vocab_size=vocab_size, max_len=max_seq_len
        )
        self.enc_layers = nn.Sequential(
            *[
                EncoderLayer(
                    d_model=d_model, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.embedding_layer(x)
        return self.enc_layers(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_outputs,
        vocab_size=10000,
        max_seq_len=5000,
        num_layers=6,
        d_model=512,
        num_heads=8,
        ff_hidden_dim=2048,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.dense = nn.Linear(d_model, num_outputs)

    def forward(self, x):
        x = self.encoder(x)
        x, _ = torch.max(x, dim=1)
        x = self.dense(x)
        return x
