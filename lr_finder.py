"""
Learning rate finder for GPT-MLX models.
Implements a learning rate finder similar to the fastai approach.
"""

import argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import time
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Callable
import math

from transformer import GPT, GPTConfig
from metrics import MetricsTracker, LearningCurveAnalyzer
from helper_function import simplify_compatible


class LRFinder:
    """Learning rate finder implementation for GPT models."""
    
    def __init__(
        self,
        model: GPT,
        data_path: str,
        batch_size: int = 4,
        context_size: int = 1024,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        stop_factor: float = 4.0,
        smoothing_factor: float = 0.05,
        log_dir: str = "lr_finder_results",
    ):
        """
        Initialize the learning rate finder.
        
        Args:
            model: GPT model
            data_path: Path to training data
            batch_size: Batch size for training
            context_size: Context size for training
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            stop_factor: Stop if loss exceeds minimum loss by this factor
            smoothing_factor: Exponential smoothing factor for loss
            log_dir: Directory to save results
        """
        self.model = model
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_size = context_size
        
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.stop_factor = stop_factor
        self.smoothing_factor = smoothing_factor
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics
        self.lrs = []
        self.losses = []
        self.raw_losses = []
        self.best_lr = None
        
        # Create function to compute loss and gradient
        self.loss_and_grad_fn = nn.value_and_grad(model, model.loss)
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Load and prepare training data."""
        from train import DataLoader
        self.data_loader = DataLoader(self.data_path, self.context_size)
        self.data_iter = self.data_loader.get_batch_iterator(self.batch_size)
        
    def _anneal_learning_rate(self, step: int) -> float:
        """
        Calculate learning rate based on exponential annealing.
        
        Args:
            step: Current step
            
        Returns:
            Learning rate for this step
        """
        if self.num_iter <= 1:
            return self.end_lr
            
        exponent = step / (self.num_iter - 1)
        return self.start_lr * (self.end_lr / self.start_lr) ** exponent
        
    def run(self) -> Tuple[float, List[float], List[float]]:
        """
        Run the learning rate finder.
        
        Returns:
            Tuple of (suggested_lr, list_of_lrs, list_of_losses)
        """
        print(f"Starting learning rate finder from {self.start_lr} to {self.end_lr}")
        
        # Track minimum loss for early stopping
        min_loss = float('inf')
        smoothed_loss = 0.0
        stop_training = False
        
        # Main training loop
        for i in range(self.num_iter):
            # Compute learning rate for this step
            lr = self._anneal_learning_rate(i)
            self.lrs.append(lr)
            
            # Get batch
            inputs, targets = next(self.data_iter)
            inputs, targets = map(mx.array, (inputs, targets))
            
            # Forward and backward pass
            loss, grads = self.loss_and_grad_fn(inputs, targets)
            loss_value = loss.item()
            self.raw_losses.append(loss_value)
            
            # Update smoothed loss
            if i == 0:
                smoothed_loss = loss_value
            else:
                smoothed_loss = smoothed_loss * (1 - self.smoothing_factor) + loss_value * self.smoothing_factor
            
            self.losses.append(smoothed_loss)
            
            # Check if we should stop (loss exploding)
            if i > 0 and smoothed_loss > min_loss * self.stop_factor:
                print(f"Stopping early at iter {i}: loss {smoothed_loss:.4f} > {min_loss:.4f} * {self.stop_factor}")
                stop_training = True
            
            # Update minimum loss
            if smoothed_loss < min_loss:
                min_loss = smoothed_loss
                
            print(f"Iter {i+1}/{self.num_iter}: lr={lr:.8f}, loss={smoothed_loss:.4f}")
            
            if stop_training:
                break
                
        # Suggest optimal learning rate
        self.best_lr = LearningCurveAnalyzer.suggest_learning_rate(self.losses, self.lrs)
        
        if self.best_lr is None:
            # Fallback: Use learning rate at minimum loss
            min_loss_idx = np.argmin(self.losses)
            self.best_lr = self.lrs[min_loss_idx] / 10.0  # Conservative estimate
            
        print(f"Learning rate finder complete. Suggested lr: {self.best_lr:.8f}")
        
        # Save results
        self._save_results()
        self._plot_results()
        
        return self.best_lr, self.lrs, self.losses
        
    def _save_results(self):
        """Save learning rate finder results to disk."""
        results = {
            "learning_rates": self.lrs,
            "smoothed_losses": self.losses,
            "raw_losses": self.raw_losses,
            "suggested_lr": self.best_lr,
            "settings": {
                "start_lr": self.start_lr,
                "end_lr": self.end_lr,
                "num_iter": self.num_iter,
                "batch_size": self.batch_size,
                "context_size": self.context_size,
            }
        }
        
        with open(os.path.join(self.log_dir, "lr_finder_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
    def _plot_results(self):
        """Plot learning rate finder results."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot loss vs learning rate (log scale)
        ax.plot(self.lrs, self.losses)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        
        # Mark suggested learning rate
        if self.best_lr is not None:
            ax.axvline(x=self.best_lr, color='r', linestyle='--', 
                      label=f'Suggested LR: {self.best_lr:.8f}')
            ax.legend()
            
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "lr_finder_plot.png"))
        plt.close()


def main():
    """Run the learning rate finder from command line."""
    parser = argparse.ArgumentParser(
        description="Run learning rate finder for GPT-MLX model"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="train.npy",
        help="Path to training data *.npy file",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "medium", "large", "xl"],
        default="small",
        help="Model size configuration",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    parser.add_argument(
        "--start_lr", 
        type=float, 
        default=1e-7,
        help="Starting learning rate",
    )
    parser.add_argument(
        "--end_lr",
        type=float,
        default=0.1,
        help="Ending learning rate",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=100,
        help="Number of iterations",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="lr_finder_results",
        help="Directory to save results",
    )
    
    args = parser.parse_args()
    
    # Create model config based on size
    model_configs = {
        "small": GPTConfig(
            n_layer=6,
            n_head=8,
            n_embd=512,
            vocab_size=50304,
            block_size=1024,
            bias=True,
        ),
        "medium": GPTConfig(
            n_layer=12,
            n_head=12,
            n_embd=768,
            vocab_size=50304,
            block_size=1024,
            bias=True,
        ),
        "large": GPTConfig(
            n_layer=24,
            n_head=16,
            n_embd=1024,
            vocab_size=50304,
            block_size=1024,
            bias=True,
        ),
        "xl": GPTConfig(
            n_layer=48,
            n_head=25,
            n_embd=1600,
            vocab_size=50304,
            block_size=1024,
            bias=True,
        ),
    }
    
    model_config = model_configs[args.model_size]
    model = GPT(model_config)
    
    from mlx.utils import tree_flatten
    mx.eval(model.parameters())
    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Created {args.model_size} model with {nparams/1e6:.2f}M parameters")
    
    # Run learning rate finder
    finder = LRFinder(
        model=model,
        data_path=args.data_path,
        batch_size=args.batch_size,
        context_size=model_config.block_size,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter,
        log_dir=args.log_dir,
    )
    
    best_lr, _, _ = finder.run()
    
    print(f"\nLearning rate finder complete.")
    print(f"Suggested learning rate: {best_lr:.8f}")
    print(f"Results saved to {args.log_dir}")


if __name__ == "__main__":
    main()