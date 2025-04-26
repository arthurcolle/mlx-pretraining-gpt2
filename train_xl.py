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


@dataclass
class TrainConfig:
    num_iters: int
    batch_size: int
    grad_accumulation_steps: int
    max_lr: float
    min_lr: float
    warmup_iters: int
    lr_decay_iters: int
    save_every: int
    early_stopping_patience: int
    validation_interval: int


class DataLoader:
    def __init__(self, data_path, context_size):
        self.data_path = data_path
        self.context_size = context_size
        self.x, self.y = self.create_training_examples()

    def create_training_examples(self):
        dataset = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        num_tokens = len(dataset)
        window_size = self.context_size + 1
        examples = num_tokens - window_size + 1

        x = np.lib.stride_tricks.as_strided(
            dataset,
            shape=(examples, window_size),
            strides=(dataset.itemsize, dataset.itemsize),
        )
        return x[:, :-1], x[:, 1:]

    def get_batch_iterator(self, batch_size):
        while True:
            perm = np.random.permutation(self.x.shape[0])

            for start in range(0, self.x.shape[0], batch_size):
                ids = perm[start : start + batch_size]

                batch_inputs = self.x[ids].astype(np.int64)
                batch_targets = self.y[ids].astype(np.int64)

                yield batch_inputs, batch_targets


class GPTTrainer:
    def __init__(
        self,
        data_path,
        val_path,
        train_config: TrainConfig,
        model_config: GPTConfig,
        checkpoint_dir=None,
    ):
        self.data_loader = DataLoader(data_path, model_config.block_size)
        self.val_loader = DataLoader(val_path, model_config.block_size) if val_path else None

        # training hyperparameters
        self.train_config = train_config
        self.batch_size = train_config.batch_size
        self.num_iters = train_config.num_iters
        self.grad_accumulation_steps = train_config.grad_accumulation_steps
        self.max_lr = train_config.max_lr
        self.min_lr = train_config.min_lr
        self.warmup_iters = train_config.warmup_iters
        self.lr_decay_iters = train_config.lr_decay_iters
        self.early_stopping_patience = train_config.early_stopping_patience
        self.validation_interval = train_config.validation_interval

        # initialize the model and optimizer
        self.model_config = model_config
        self.model = GPT(model_config)
        self.optimizer = opt.AdamW(learning_rate=self.max_lr)
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self.model.loss)

        # init gradient accumulation state
        self.accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), self.model.parameters()
        )
        self.accumulated_loss = 0.0
        self.iter_num = 0

        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.checkpoint_dir = checkpoint_dir
        self.data_path = data_path
        self.val_path = val_path
        self.data_loader = DataLoader(data_path, self.model_config.block_size)

        # Initialize plots directory
        self.plots_dir = os.path.join(checkpoint_dir, "plots")
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def save_checkpoint(self, checkpoint_dir, is_best=False):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        file_prefix = "best_model" if is_best else "model"
        model_weights_path = os.path.join(checkpoint_dir, f"{file_prefix}_weights.npz")
        model_config_path = os.path.join(checkpoint_dir, "model_config.json")

        self.model.save_weights(model_weights_path)

        with open(model_config_path, "w") as f:
            json.dump(self.model_config.__dict__, f)

        # Save training history
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "iterations": list(range(1, len(self.train_losses) + 1)),
            "current_iteration": self.iter_num,
            "best_val_loss": self.best_val_loss,
        }
        with open(history_path, "w") as f:
            json.dump(history, f)

    def print_parameter_count(self):
        mx.eval(self.model.parameters())
        nparams = sum(x.size for k, x in tree_flatten(self.model.parameters()))
        print(f"Training a custom GPT model with {nparams / 1e6:.3f} M parameters")

    def print_loss(self, iteration_count, average_loss, tic):
        toc = time.perf_counter()
        print(
            f"iter {iteration_count}: train loss {average_loss:.3f}, "
            f"it/sec {1.0 / (toc - tic):.3f}, "
            f"lr {self.optimizer.learning_rate:.4f}"
        )
        return toc

    def update_learning_rate(self, it):
        if it < self.warmup_iters:
            new_lr = self.max_lr * it / self.warmup_iters
        elif it > self.lr_decay_iters:
            new_lr = self.min_lr
        else:
            decay_ratio = (it - self.warmup_iters) / (
                self.lr_decay_iters - self.warmup_iters
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            new_lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        self.optimizer.set_learning_rate(new_lr)

    def compute_minibatch_loss_grads(self, inputs, targets):
        inputs, targets = map(mx.array, (inputs, targets))
        loss, grads = self.loss_and_grad_fn(inputs, targets)

        self.accumulated_grads = tree_map(
            lambda acc, new: acc + new * (1.0 / self.grad_accumulation_steps),
            self.accumulated_grads,
            grads,
        )

        tree_map(
            lambda grad: mx.eval(grad),
            self.accumulated_grads,
        )

        self.accumulated_loss += loss.item()
        return loss

    def compute_batch_loss(self, loss):
        average_loss = self.accumulated_loss / self.grad_accumulation_steps
        self.accumulated_loss = 0.0
        mx.simplify(loss, self.model.parameters())
        mx.eval(loss, self.model.parameters())
        return average_loss

    def perform_gradient_step(self):
        self.model.update(
            self.optimizer.apply_gradients(self.accumulated_grads, self.model)
        )
        self.accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), self.model.parameters()
        )

    def evaluate(self, num_batches=5):
        """Evaluate the model on validation data"""
        if self.val_loader is None:
            return None
        
        val_data = self.val_loader.get_batch_iterator(self.batch_size)
        val_loss = 0.0
        
        for _ in range(num_batches):
            inputs, targets = next(val_data)
            inputs, targets = map(mx.array, (inputs, targets))
            logits = self.model(inputs)
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            val_loss += mx.mean(loss).item()
            
        return val_loss / num_batches

    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        
        if self.val_losses:
            # Create x-axis values for validation points (every validation_interval)
            val_x = list(range(0, len(self.train_losses), self.validation_interval))
            # Make sure we have the right number of points
            val_y = self.val_losses[:len(val_x)]
            plt.plot(val_x, val_y, 'r', label='Validation Loss')
        
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss (GPT-2 XL)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f"loss_plot_iter_{self.iter_num}.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Loss plot saved to {plot_path}")

    def train(self):
        self.print_parameter_count()

        tic = time.perf_counter()
        train_data = self.data_loader.get_batch_iterator(self.batch_size)

        for iteration in range(self.num_iters * self.grad_accumulation_steps):
            inputs, targets = next(train_data)
            loss = self.compute_minibatch_loss_grads(inputs, targets)

            if (iteration + 1) % self.grad_accumulation_steps == 0:
                self.perform_gradient_step()
                self.update_learning_rate(self.iter_num)
                batch_loss = self.compute_batch_loss(loss)
                self.train_losses.append(batch_loss)
                tic = self.print_loss(self.iter_num, batch_loss, tic)

                # Validation
                if self.val_loader and self.iter_num % self.validation_interval == 0:
                    val_loss = self.evaluate()
                    self.val_losses.append(val_loss)
                    print(f"iter {self.iter_num}: val loss {val_loss:.3f}")
                    
                    # Early stopping
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        print(f"New best validation loss: {val_loss:.3f}")
                        self.save_checkpoint(self.checkpoint_dir, is_best=True)
                    else:
                        self.patience_counter += 1
                        print(f"Validation loss did not improve. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                        
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"Early stopping triggered after {self.iter_num} iterations")
                            # Plot final losses
                            self.plot_losses()
                            return

                # Checkpoint saving
                if self.iter_num % self.train_config.save_every == 0:
                    print(f"Saving model to {self.checkpoint_dir}")
                    self.save_checkpoint(self.checkpoint_dir)
                    self.plot_losses()

                self.iter_num += 1

        # Plot final losses
        self.plot_losses()


def main(train_path, val_path, checkpoint_dir):
    # GPT-2 XL (1.5B params) configuration
    model_config = GPTConfig(
        n_layer=48,
        n_head=25,
        n_embd=1600,
        vocab_size=50304,
        block_size=1024,
        bias=True,
    )

    train_config = TrainConfig(
        num_iters=200,
        batch_size=1,
        grad_accumulation_steps=32,  # Increased for large model
        max_lr=1e-4,
        min_lr=1e-5,
        warmup_iters=20,
        lr_decay_iters=200,
        save_every=10,
        early_stopping_patience=5,
        validation_interval=10,
    )
    trainer = GPTTrainer(train_path, val_path, train_config, model_config, checkpoint_dir)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 XL (1.5B parameter) model on a custom dataset"
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
        default="checkpoints_xl",
        help="Path to checkpoint directory to save model weights",
    )
    args = parser.parse_args()

    main(args.data_path, args.val_path, args.checkpoint_dir)