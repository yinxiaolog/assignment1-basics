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
        weight = torch.empty(self.d_out, self.d_in)
        std = (2 / (self.d_in + self.d_out)) ** 0.5
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3, b=3)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


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
        self.weight = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        weight = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        rms = torch.sum(x**2, -1, keepdim=True) / self.d_model + self.eps
        rms = torch.sqrt(rms)
        result = x / rms * self.weight
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

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
        cos_rotation = []
        sin_rotation = []
        for i in range(max_seq_len):
            cos_theta, sin_theta = self.theta_vec(i, d_k, theta)
            cos_rotation.append(cos_theta)
            sin_rotation.append(sin_theta)
        self.cos_rotation = torch.stack(cos_rotation)
        self.sin_rotation = torch.stack(sin_rotation)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        q = x
        p = q.clone()
        n = x.shape[-1]
        indices = torch.arange(n)
        indices[::2] = torch.arange(1, n, 2)
        indices[1::2] = torch.arange(0, n, 2)
        p = p.index_select(-1, indices)
        p[..., torch.arange(0, n, 2)] *= -1
        return (
            q * self.cos_rotation[: x.shape[-2]] + p * self.sin_rotation[: x.shape[-2]]
        )

    def theta_vec(self, m: int, d: int, theta: float):
        thetas = []
        for i in range(d // 2):
            thetas.append(m / theta ** (2 * i / d))
            thetas.append(m / theta ** (2 * i / d))
        thetas = torch.tensor(thetas, dtype=torch.float32)
        return torch.cos(thetas), torch.sin(thetas)


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
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        is_apply_rope=False,
        max_seq_len=1024,
        theta=10000,
    ):
        super().__init__()
        self.is_apply_rope = is_apply_rope
        self.num_heads = num_heads
        assert (
            d_model % self.num_heads == 0
        ), f"d_model does not match num_heads, d_model={d_model}, num_heads={self.num_heads}"
        self.head_dim = d_model // num_heads
        self.attention = Attention()
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = RoPE(theta, self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.tril(torch.ones(x.shape[1], x.shape[1])).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).tile(x.shape[0], self.num_heads, 1, 1)
        q = (
            self.q_proj(x)
            .reshape(-1, x.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .reshape(-1, x.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(-1, x.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.is_apply_rope:
            q = self.rope(q, None)
            k = self.rope(k, None)
        atten = self.attention(q, k, v, mask).transpose(1, 2).reshape(x.shape)
        return self.output_proj(atten)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: 1024, theta: 0.1
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, True, max_seq_len, theta
        )
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.ln1(x))
        y = y + self.ffn(self.ln2(y))
        return y


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: 0.1,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        blocks = [
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_layers)
        ]
        self.layers = nn.Sequential(*blocks)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.softmax = Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
