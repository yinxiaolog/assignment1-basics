import os
import math
import json
from typing import Optional
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
import swanlab
from .bpe import Tokenizer


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


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device
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
        indices = torch.arange(n, device=x.device)
        self.cos_rotation = self.cos_rotation.to(device=x.device)
        self.sin_rotation = self.sin_rotation.to(device=x.device)
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
        mask = mask.to(device=q.device)
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
        theta: 10000,
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


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input -= torch.max(input, dim=-1, keepdim=True).values
        loss = (
            input.gather(-1, target.reshape(-1, 1))
            - torch.sum(torch.exp(input), dim=-1, keepdim=True).log()
        )
        return -loss.sum() / input.shape[0]


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8):
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                m = state.get("m", torch.zeros(p.grad.shape, device=p.device))
                v = state.get("v", torch.zeros(p.grad.shape, device=p.device))
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)
                lr_t = lr * (((1 - beta2**t) ** 0.5) / (1 - beta1**t))
                p.data -= lr_t * m / (v**0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate

    return min_learning_rate + 0.5 * (
        1
        + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
    ) * (max_learning_rate - min_learning_rate)


class CosineLR(LRScheduler):
    def __init__(
        self,
        optimizer,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
        last_epoch=-1,
        verbose="deprecated",
    ):

        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        it = self.last_epoch

        if it < self.warmup_iters:
            lr = it / self.warmup_iters * self.max_learning_rate
        elif it > self.cosine_cycle_iters:
            lr = self.min_learning_rate
        else:
            lr = self.min_learning_rate + 0.5 * (
                1
                + math.cos(
                    (it - self.warmup_iters)
                    / (self.cosine_cycle_iters - self.warmup_iters)
                    * math.pi
                )
            ) * (self.max_learning_rate - self.min_learning_rate)
        return [lr for _ in self.optimizer.param_groups]


def gradient_clipping(params, max_norm):
    grads = [param.grad for param in params if param.grad is not None]
    grads = torch.stack(grads).reshape(-1)
    l2 = torch.norm(grads, p=2)
    if l2 >= max_norm:
        alpha = max_norm / (l2 + 1e-6)
        for param in params:
            if param.grad is not None:
                param.grad.mul_(alpha)


class LMDataset(Dataset):
    def __init__(self, data, context_length, stride=1, device="cpu"):
        super().__init__()
        self.data = data
        self.context_length = context_length
        self.stride = stride
        self.device = device

    def __len__(self):
        return (len(self.data) - 1 - self.context_length) // self.stride + 1
        # return len(self.data) - self.context_length

    def __getitem__(self, index):
        index *= self.stride
        input = self.data[index : index + self.context_length]
        label = self.data[index + 1 : index + 1 + self.context_length]
        return torch.tensor(input, dtype=torch.long).to(self.device), torch.tensor(
            label, dtype=torch.long
        ).to(self.device)


class LMDataLoader(DataLoader): ...


def save_checkpoint(model, optimizer, iteration, path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        path,
    )


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


class Config:
    epochs = 5
    lr = 1e-4
    batch_size = 1
    vocab_size = 50257
    context_length = 1024
    device = "cpu"
    loss_fn = CrossEntropyLoss()
    optim = None

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4, default=str, ensure_ascii=False)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        test_dataset: Dataset = None,
        config: Config = None,
        project: str = "default",
        experiment_name="foo",
        description: str = "default",
        total_tokens_processed=327680000,
    ):
        self.swanlab_run = swanlab.init(
            project=project, experiment_name=experiment_name, description=description
        )

        swanlab.config = {"cfg": config}
        print(f"cfg: {config}")
        self.device = config.model.device
        self.model: nn.Module = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=config.model.batch_size, shuffle=True
        )
        self.val_dataloader = (
            DataLoader(val_dataset, batch_size=config.model.batch_size * 4, shuffle=False)
            if val_dataset is not None
            else None
        )
        self.test_dataloader = (
            DataLoader(test_dataset, batch_size=config.model.batch_size, shuffle=True)
            if test_dataset is not None
            else None
        )
        self.config = config
        if config.loss_fn.name == "CrossEntropyLoss":
            self.loss_fn = CrossEntropyLoss()
        else:
            raise Exception(f"not support loss_fn: {config.loss_fn.name}")
        
        if config.optimizer.name == "AdamW":
            self.optim = AdamW(self.model.parameters(), lr=config.optimizer.lr)
        else:
            raise Exception(f"not support optimizer: {config.loss_fn.name}")
        #self.scheduler = CosineLR(self.optim, config.optimizer.lr, 1e-5, warmup_iters=1000, cosine_cycle_iters=30000)
        self.total_tokens_processed = total_tokens_processed

    def train(self, checkpoint_path=None):
        step = 0
        if checkpoint_path is not None:
            step = load_checkpoint(checkpoint_path, self.model, self.optim)
        self.model.train()
        
        for epoch in range(self.config.model.epochs):
            for x, y in self.train_dataloader:
                loss = self.train_one_step(x, y)
                step += 1
                self.swanlab_run.log({"train loss": loss})
                self.swanlab_run.log({"process": self.config.model.batch_size * step * self.config.model.context_length / self.total_tokens_processed * 100})
                if (
                    self.total_tokens_processed > 0
                    and self.config.model.batch_size * step * self.config.model.context_length
                    >= self.total_tokens_processed
                ):
                    exit(0)
                if step % 1000 == 0:
                    print(
                        f"step: {step}\n story: {self.test("The rain had just stopped when Emma stepped off the train.")}"
                    )
                    save_checkpoint(
                        self.model,
                        self.optim,
                        step,
                        path=os.path.join(
                            self.config.log.dir, f"model_optim_step_{step}.pth"
                        ),
                    )
                if step % 10 == 0:
                    print(
                        f"epoch: {epoch} step: {step} process: {self.config.model.batch_size * step * self.config.model.context_length / self.total_tokens_processed:.2%}, loss={loss:.3f}"
                    )
                if step % 100 == 0:
                    val_loss = self.val()
                    print(f"val loss: {val_loss}")
                    self.swanlab_run.log({"val loss:": val_loss})

    def train_one_step(self, x: torch.Tensor, label: torch.Tensor) -> float:
        x = x.to(self.device)
        label = label.to(self.device)
        y = self.model(x)
        loss = self.loss_fn(y.reshape(-1, y.shape[-1]), label.reshape(-1))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        #self.scheduler.step()
        return loss.item()

    @torch.inference_mode()
    def val(self):
        if self.val_dataloader is None:
            return 0
        self.model.eval()
        device = self.device
        all_loss = 0
        size = len(self.val_dataloader.dataset) / self.config.model.batch_size
        i = 0
        # print(self.config.batch_size)
        for x, label in self.val_dataloader:
            x = x.to(device)
            label = label.to(device)
            y = self.model(x)
            loss = self.loss_fn(y.reshape(-1, y.shape[-1]), label.reshape(-1))
            # print(f"{i} / {size}: loss={loss}")
            all_loss += loss.item() * len(label)
            i += 1
        return all_loss / len(self.val_dataloader.dataset)

    def test(
        self,
        prompt: str,
        max_new_token: int = 1000,
        top_k=None,
        temperature: float = 0.0,
    ):
        self.model.eval()
        idx = self.tokenizer.encode(prompt)
        idx = torch.tensor(idx, device=self.device)
        idx = idx.unsqueeze(0)
        for _ in range(max_new_token):
            input = idx[:, -self.config.model.context_length :]
            logits = self.model(input)[:, -1, :]
            if top_k is not None:
                top_k_logits, _ = torch.topk(logits, top_k)
                min_top_k = top_k_logits[:, -1]
                logits = torch.where(
                    logits < min_top_k,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            if temperature > 0:
                logits = logits / temperature
                prob = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(prob, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if (
                idx_next.item()
                == self.tokenizer.token_2_id["<|endoftext|>".encode(encoding="utf-8")]
            ):
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return self.tokenizer.decode(idx.reshape(-1).tolist())


class Inference:
    def __init__(self, model: nn.Module, state_dict, tokenizer: Tokenizer):
        model.load_state_dict(state_dict)
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.context_length = 1024

    def run(
        self,
        prompt: str,
        max_new_token: int = 1000,
        top_k=None,
        temperature: float = 0.0,
    ):
        self.model.eval()
        idx = self.tokenizer.encode(prompt)
        idx = torch.tensor(idx, device=self.device)
        idx = idx.unsqueeze(0)
        with torch.inference_mode():
            for _ in range(max_new_token):
                input = idx[:, -self.context_length :]
                logits = self.model(input)[:, -1, :]
                if top_k is not None:
                    top_k_logits, _ = torch.topk(logits, top_k)
                    min_top_k = top_k_logits[:, -1]
                    logits = torch.where(
                        logits < min_top_k,
                        torch.tensor(float("-inf")).to(logits.device),
                        logits,
                    )

                if temperature > 0:
                    logits = logits / temperature
                    prob = torch.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(prob, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                if (
                    idx_next.item()
                    == self.tokenizer.token_2_id[
                        "<|endoftext|>".encode(encoding="utf-8")
                    ]
                ):
                    pass
                idx = torch.cat((idx, idx_next), dim=1)
        return self.tokenizer.decode(idx.reshape(-1).tolist())
