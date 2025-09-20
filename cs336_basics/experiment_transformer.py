import json
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

from .mynn import (
    TransformerLM,
    SGD,
    Config,
    Trainer,
    LMDataset,
    LMDataLoader,
    AdamW,
    Inference,
    CosineLR,
)
from .bpe import load_tokenizer, Tokenizer
from .config import CudaConfig


def transformer_accounting(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
):
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=10000,
    )

    params_account = sum(param.numel() for param in model.parameters())
    print(
        f"parameters: {params_account / 1000000}M memory: {params_account * 4 / 1024 / 1024} MB"
    )

    qkvo_proj = 8 * context_length * d_model * d_model
    attention = (
        2 * context_length * context_length * d_model
        + 2 * context_length * context_length * d_model
    )
    ffn = 4 * context_length * d_model * d_ff + 2 * context_length * d_model * d_ff
    final_linear = 2 * context_length * d_model * vocab_size

    flops = {
        "multiHeadAttention": {
            "qkvProj": qkvo_proj * num_layers / 1000000,
            "attention": attention * num_layers / 1000000,
        },
        "ffn": ffn * num_layers / 1000000,
        "finalLinear": final_linear / 1000000,
    }

    print(json.dumps(flops, indent=4))
    all_flops = (qkvo_proj + attention + ffn) * num_layers + final_linear
    print(
        f"multiHeadAttention: {(qkvo_proj + attention) * num_layers / all_flops:.2f}, ffn: {ffn * num_layers / all_flops:.2f}, final_linear: {final_linear / all_flops:.2f}, all_flops: {all_flops}"
    )


def flops_accounting_gpt2():
    # GPT2 small
    print("====================GPT2 small====================")
    transformer_accounting(
        vocab_size=50257,
        context_length=1024,
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=6400,
    )

    # GPT2 medium
    print("====================GPT2 medium==================")
    transformer_accounting(
        vocab_size=50257,
        context_length=1024,
        num_layers=24,
        d_model=1024,
        num_heads=16,
        d_ff=6400,
    )

    # GPT2 large
    print("====================GPT2 large===================")
    transformer_accounting(
        vocab_size=50257,
        context_length=1024,
        num_layers=36,
        d_model=1280,
        num_heads=20,
        d_ff=6400,
    )

    # GPT2 XL
    print("====================GPT2 Xl====================")
    transformer_accounting(
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
    )


def learning_rate_tuning(lr, iterations=10):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    for t in range(iterations):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def training_loop(cfg: DictConfig):
    print(cfg)
    train_dataset = LMDataset(
        data=np.load(file=cfg.data.train_dataset_path, mmap_mode="r"),
        context_length=cfg.model.context_length,
    )
    val_dataset = LMDataset(
        data=np.load(file=cfg.data.val_dataset_path, mmap_mode="r"),
        context_length=cfg.model.context_length,
        stride=cfg.model.context_length,
    )

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        theta=10000,
    )
    model = torch.compile(model, backend=cfg.model.backend)
    model = model.to(cfg.model.device)
    tokenizer: Tokenizer = load_tokenizer(cfg.data.tokenizer_path)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg,
        project="cs336_basics",
        experiment_name=f"train_tinystory lr={cfg.optimizer.lr}, batch_size={cfg.model.batch_size}",
        description="training tinystory",
    )
    trainer.train()


def inferance():
    model = TransformerLM(
        vocab_size=50257,
        context_length=1024,
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=6400,
        theta=10000,
    )
    tokenizer: Tokenizer = load_tokenizer(
        "/Users/yinxiaoloong/dataset/cs336/TinyStoriesV2-GPT4-train_tokenzier.pkl"
    )
    infer = Inference(
        model,
        torch.load("/Users/yinxiaoloong/log/model_optim_step_10000.pth")["model"],
        tokenizer,
    )
    print(
        infer.run(
            """u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend."""
        )
    )


if __name__ == "__main__":
    # print(len(np.load(file="/opt/code/cs336/assignment1-basics/ts_train.npy", mmap_mode="r")))
    training_loop()
    # inferance()
