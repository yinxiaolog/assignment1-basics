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


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        # print(f"d_k: {d_k}")
        # theta_range = torch.arange(d_k, device=device)
        # for i in range(0, d_k, 2):
        #     theta_range[i] = i / d_k
        #     theta_range[i] = i / d_k
        # theta_range = torch.pow(theta, -theta_range)
        # cos_rotation = []
        # sin_rotation = []
        # for i in range(0, 2 * max_seq_len, 2):
        #     cos_theta = torch.cos(i * theta_range)
        #     sin_theta = torch.sin(i * theta_range)
        #     cos_rotation.append(cos_theta)
        #     sin_rotation.append(sin_theta)
        # cos_rotation = torch.stack(cos_rotation)
        # sin_rotation = torch.stack(sin_rotation)
        # self.cos_rotation = cos_rotation
        # self.sin_rotation = sin_rotation
        rotation = []
        for i in range(max_seq_len):
            rotation.append(self.r_i(i, d_k, theta))
        self.rotation = rotation

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x_copy = x.clone()
        # x_copy[..., 1::2] *= -1
        # indices = torch.arange(x.shape[-1])
        # indices[::2] += 1
        # indices[1::2] -= 1
        # sequence_length = token_positions.shape[-1]
        # cos_rotation = self.cos_rotation[: sequence_length,:]
        # sin_rotation = self.sin_rotation[: sequence_length,:]
        # x_copy = x_copy[..., indices]
        # print(x.shape, self.cos_rotation.shape)
        # return x * cos_rotation + x_copy * sin_rotation
        out = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out.append(x[i][j] @ self.rotation[j].T)
        return torch.stack(out).reshape(x.shape)

    def r_ik(self, i: int, k: int, d: int, theta: float) -> torch.Tensor:
        theta = i / theta ** (2 * k / d)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        return torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    
    def r_i(self, i: int, d: int, theta: float) -> torch.Tensor:
        r = torch.zeros(d, d, dtype=torch.float32)
        for k in range(0, d, 2):
            r[k: k + 2, k: k + 2] = self.r_ik(i, k / 2, d, theta)
        return r
    
    
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
