import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import optimizer as opt
import math
import os
import json
import matplotlib.pyplot as plt

from mlx.utils import tree_flatten, tree_map
from transformer import GPT, GPTConfig
from dataclasses import dataclass
from metrics import MetricsTracker, PerplexityCalculator
from sample_generator import TextSampleGenerator
from helper_function import simplify_compatible
from train import DataLoader, TrainConfig, GPTTrainer


def main(train_path, val_path, checkpoint_dir, use_wandb=False, use_tensorboard=True):
    # Small model config (6-layer model with fewer parameters)
    model_config = GPTConfig(
        n_layer=6,
        n_head=8,
        n_embd=512,
        vocab_size=50304,
        block_size=512,
        bias=True,
    )

    train_config = TrainConfig(
        num_iters=100,
        batch_size=4,
        grad_accumulation_steps=2,
        max_lr=1.8e-3,  # Based on LR finder
        min_lr=1.8e-4,
        warmup_iters=10,
        lr_decay_iters=90,
        save_every=10,
        early_stopping_patience=5,
        validation_interval=5,
        sample_every=10,
        num_samples=3,
        max_sample_tokens=100,
    )
    
    # Sample prompts for text generation during training
    sample_prompts = [
        "Once upon a time",
        "In a world where",
        "The future of AI is"
    ]
    
    trainer = GPTTrainer(
        train_path, 
        val_path, 
        train_config, 
        model_config, 
        checkpoint_dir,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        sample_prompts=sample_prompts
    )
    
    # Print model info
    nparams = sum(x.size for k, x in tree_flatten(trainer.model.parameters()))
    print(f"Training a small GPT model with {nparams / 1e6:.3f}M parameters")
    print(f"Context length: {model_config.block_size}")
    print(f"Number of layers: {model_config.n_layer}")
    print(f"Number of heads: {model_config.n_head}")
    print(f"Embedding dimension: {model_config.n_embd}")
    
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a small GPT model on a custom dataset"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="train.npy",
        help="Path to training data *.npy file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="val.npy",
        help="Path to validation data *.npy file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints_small",
        help="Path to checkpoint directory to save model weights",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for tracking",
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        default=True,
        help="Whether to use TensorBoard for tracking",
    )
    args = parser.parse_args()

    main(args.data_path, args.val_path, args.checkpoint_dir, 
         use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard)