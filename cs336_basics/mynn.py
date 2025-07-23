import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.device = device
        self.dtype = dtype
        weights = torch.empty(self.d_out, self.d_in)
        std = (2 / (self.d_in + self.d_out)) ** 0.5
        nn.init.trunc_normal_(weights, mean=0, std=std, a=-3, b=3)
        self.weights = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        embedding_matrix = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )
        nn.init.trunc_normal_(embedding_matrix, mean=0, std=1, a=-3, b=3)
        self.embedding_matrix = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        g = torch.ones(d_model, device=device, dtype=dtype)
        self.g = nn.Parameter(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        rms = torch.sum(x**2, -1, keepdim=True) / self.d_model + self.eps
        rms = torch.sqrt(rms)
        result = x / rms * self.g
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        w1 = torch.empty(d_model, d_ff, device=device, dtype=dtype)
        w2 = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        w3 = torch.empty(d_model, d_ff, device=device, dtype=dtype)
        nn.init.trunc_normal_(w1)
        nn.init.trunc_normal_(w2)
        nn.init.trunc_normal_(w3)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        x_max = torch.max(x, dim=dim, keepdim=True).values
        out = x - x_max
        out = torch.exp(out)
        return out / torch.sum(out, dim=dim, keepdim=True)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = q @ k.mT
        out = out / (k.shape[-1] ** 0.5)
        if mask is not None:
            out.masked_fill_(~mask, -float("inf"))
        out = self.softmax(out, dim=-1)
        return out @ v


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = Attention()
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        mask = torch.tril(torch.ones(d_model, d_model)).bool()
        self.mask = mask


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v
        atten = self.attention(q, k, v, self.mask)
        return atten @ self.w_o
