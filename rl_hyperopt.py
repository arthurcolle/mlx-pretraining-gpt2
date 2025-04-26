import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import math
import os
import json
import random
import psutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from transformer import GPT, GPTConfig
from optimizer import AdamW
from helper_function import simplify_compatible
from mlx.utils import tree_flatten, tree_map

global MAX_MEMORY_GB

# Memory constraint in GB
MAX_MEMORY_GB = 20


@dataclass
class HyperParams:
    """Hyperparameters that can be dynamically adjusted during training"""
    learning_rate: float
    batch_size: int
    grad_accumulation_steps: int
    dropout: float
    weight_decay: float


@dataclass
class TrainConfig:
    """Base training configuration"""
    num_iters: int
    max_lr: float
    min_lr: float
    warmup_iters: int
    lr_decay_iters: int
    save_every: int
    rl_update_freq: int  # How often to update hyperparams using RL
    exploration_factor: float  # Controls exploration vs exploitation


class DataLoader:
    def __init__(self, data_path, context_size, memory_efficient=True):
        self.data_path = data_path
        self.context_size = context_size
        self.memory_efficient = memory_efficient
        
        # Use memory-efficient loading if enabled
        if memory_efficient:
            self.dataset = np.memmap(self.data_path, dtype=np.uint16, mode="r")
            self.num_tokens = len(self.dataset)
            self.window_size = self.context_size + 1
            self.examples = self.num_tokens - self.window_size + 1
            self.x = None
            self.y = None
            print(f"Using memory-efficient data loading (streaming instead of preloading)")
        else:
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
    
    def get_random_batch(self, batch_size):
        """Memory-efficient random batch sampling"""
        if not self.memory_efficient:
            # Use preloaded data if memory efficiency is off
            perm = np.random.permutation(self.x.shape[0])
            ids = perm[:batch_size]
            batch_inputs = self.x[ids].astype(np.int64)
            batch_targets = self.y[ids].astype(np.int64)
            return batch_inputs, batch_targets
            
        # Generate random starting indices
        indices = np.random.randint(0, self.examples, size=batch_size)
        batch_inputs = np.zeros((batch_size, self.context_size), dtype=np.int64)
        batch_targets = np.zeros((batch_size, self.context_size), dtype=np.int64)
        
        # Extract sequences for each index
        for i, idx in enumerate(indices):
            batch_inputs[i] = self.dataset[idx:idx+self.context_size]
            batch_targets[i] = self.dataset[idx+1:idx+self.context_size+1]
            
        return batch_inputs, batch_targets

    def get_batch_iterator(self, batch_size):
        while True:
            if self.memory_efficient:
                # For memory-efficient mode, just return random batches
                yield self.get_random_batch(batch_size)
            else:
                # Original implementation for non-memory-efficient mode
                perm = np.random.permutation(self.x.shape[0])
    
                for start in range(0, self.x.shape[0], batch_size):
                    ids = perm[start : start + batch_size]
    
                    batch_inputs = self.x[ids].astype(np.int64)
                    batch_targets = self.y[ids].astype(np.int64)
    
                    yield batch_inputs, batch_targets


class RLHyperOptimizer:
    """
    Reinforcement Learning-based Hyperparameter Optimizer
    Uses a simple multi-armed bandit approach with epsilon-greedy exploration
    """
    def __init__(self, 
                 base_hyperparams: HyperParams,
                 exploration_factor: float = 0.3,
                 reward_window: int = 5):
        self.base_hyperparams = base_hyperparams
        self.exploration_factor = exploration_factor
        self.reward_window = reward_window
        
        self.hyperparams_history = []
        self.reward_history = []
        self.recent_losses = []
        
        # Parameter ranges for exploration
        self.param_ranges = {
            "learning_rate": (base_hyperparams.learning_rate * 0.5, base_hyperparams.learning_rate * 2.0),
            "batch_size": (max(1, base_hyperparams.batch_size - 1), base_hyperparams.batch_size + 1),
            "grad_accumulation_steps": (max(1, base_hyperparams.grad_accumulation_steps - 4), 
                                       base_hyperparams.grad_accumulation_steps + 4),
            "dropout": (max(0.0, base_hyperparams.dropout - 0.1), min(0.5, base_hyperparams.dropout + 0.1)),
            "weight_decay": (max(0.0, base_hyperparams.weight_decay - 0.005), 
                            base_hyperparams.weight_decay + 0.005)
        }
        
        # Initialize with base hyperparameters
        self.current_hyperparams = base_hyperparams
        self.best_hyperparams = base_hyperparams
        self.best_reward = float('-inf')

    def update_loss(self, loss: float):
        """Record the latest loss value"""
        self.recent_losses.append(loss)
        if len(self.recent_losses) > self.reward_window:
            self.recent_losses.pop(0)
    
    def compute_reward(self) -> float:
        """Compute reward based on improvement in loss and stability"""
        if len(self.recent_losses) < self.reward_window:
            return 0.0
        
        # Lower loss is better, so negative average
        avg_loss = np.mean(self.recent_losses)
        loss_std = np.std(self.recent_losses)
        
        # Reward is negative loss with a penalty for high variance
        reward = -avg_loss - 0.1 * loss_std
        return reward
    
    def update(self) -> HyperParams:
        """Update hyperparameters using RL strategy"""
        # Compute reward for current hyperparameters
        reward = self.compute_reward()
        self.reward_history.append(reward)
        self.hyperparams_history.append(self.current_hyperparams)
        
        # Update best hyperparameters if current is better
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_hyperparams = self.current_hyperparams
        
        # Decide between exploration and exploitation
        if random.random() < self.exploration_factor:
            # Exploration: randomly adjust one hyperparameter
            new_hyperparams = self._explore()
        else:
            # Exploitation: use best hyperparameters with small random adjustment
            new_hyperparams = self._exploit()
        
        self.current_hyperparams = new_hyperparams
        return new_hyperparams
    
    def _explore(self) -> HyperParams:
        """Random exploration of hyperparameter space"""
        # Copy current hyperparameters
        new_hyperparams = HyperParams(
            learning_rate=self.current_hyperparams.learning_rate,
            batch_size=self.current_hyperparams.batch_size,
            grad_accumulation_steps=self.current_hyperparams.grad_accumulation_steps,
            dropout=self.current_hyperparams.dropout,
            weight_decay=self.current_hyperparams.weight_decay
        )
        
        # Choose a random parameter to adjust
        param_to_change = random.choice(list(self.param_ranges.keys()))
        
        # Adjust the chosen parameter within its range
        if param_to_change == "learning_rate":
            min_val, max_val = self.param_ranges["learning_rate"]
            new_hyperparams.learning_rate = random.uniform(min_val, max_val)
        elif param_to_change == "batch_size":
            min_val, max_val = self.param_ranges["batch_size"]
            new_hyperparams.batch_size = int(random.uniform(min_val, max_val))
        elif param_to_change == "grad_accumulation_steps":
            min_val, max_val = self.param_ranges["grad_accumulation_steps"]
            new_hyperparams.grad_accumulation_steps = int(random.uniform(min_val, max_val))
        elif param_to_change == "dropout":
            min_val, max_val = self.param_ranges["dropout"]
            new_hyperparams.dropout = random.uniform(min_val, max_val)
        elif param_to_change == "weight_decay":
            min_val, max_val = self.param_ranges["weight_decay"]
            new_hyperparams.weight_decay = random.uniform(min_val, max_val)
        
        return new_hyperparams
    
    def _exploit(self) -> HyperParams:
        """Exploitation with small random adjustments"""
        # Start with best hyperparameters
        new_hyperparams = HyperParams(
            learning_rate=self.best_hyperparams.learning_rate,
            batch_size=self.best_hyperparams.batch_size,
            grad_accumulation_steps=self.best_hyperparams.grad_accumulation_steps,
            dropout=self.best_hyperparams.dropout,
            weight_decay=self.best_hyperparams.weight_decay
        )
        
        # Add small random adjustments (10% of the way toward exploration)
        exploration_factor = 0.1
        for param in self.param_ranges.keys():
            if param == "learning_rate":
                min_val, max_val = self.param_ranges["learning_rate"]
                deviation = (max_val - min_val) * exploration_factor
                new_hyperparams.learning_rate += random.uniform(-deviation, deviation)
                new_hyperparams.learning_rate = max(min_val, min(max_val, new_hyperparams.learning_rate))
            elif param == "batch_size":
                # Keep batch size mostly stable during exploitation
                pass
            elif param == "grad_accumulation_steps":
                # Keep grad accumulation mostly stable during exploitation
                pass
            elif param == "dropout":
                min_val, max_val = self.param_ranges["dropout"]
                deviation = (max_val - min_val) * exploration_factor
                new_hyperparams.dropout += random.uniform(-deviation, deviation)
                new_hyperparams.dropout = max(min_val, min(max_val, new_hyperparams.dropout))
            elif param == "weight_decay":
                min_val, max_val = self.param_ranges["weight_decay"]
                deviation = (max_val - min_val) * exploration_factor
                new_hyperparams.weight_decay += random.uniform(-deviation, deviation)
                new_hyperparams.weight_decay = max(min_val, min(max_val, new_hyperparams.weight_decay))
        
        return new_hyperparams


class ForwardForwardTrainer:
    """
    Implementation of the forward-forward algorithm for unsupervised learning
    as proposed by Geoffrey Hinton.
    
    This is a simplified version that alternates between:
    1. Regular language model training (negative phase)
    2. Forward-forward unsupervised training (positive phase)
    """
    def __init__(self, model: GPT, data_loader, ff_update_freq: int = 10):
        self.model = model
        self.data_loader = data_loader
        self.ff_update_freq = ff_update_freq
        
        # Threshold for positive/negative examples
        self.threshold = 2.0
        
    def forward_forward_step(self, inputs):
        """
        Perform a forward-forward algorithm step
        Instead of using the loss from prediction, we use the goodness of fit
        """
        # Get positive samples (real data)
        positive_inputs = inputs
        
        # Create negative samples by randomly permuting the inputs
        negative_indices = np.random.permutation(inputs.shape[1])
        negative_inputs = inputs[:, negative_indices]
        
        # Forward pass for positive samples
        pos_goodness = self._compute_goodness(positive_inputs)
        
        # Forward pass for negative samples  
        neg_goodness = self._compute_goodness(negative_inputs)
        
        # Loss is based on pushing positive samples above threshold and negative below
        pos_loss = mx.mean(mx.relu(self.threshold - pos_goodness))
        neg_loss = mx.mean(mx.relu(neg_goodness - self.threshold))
        
        # Total loss
        ff_loss = pos_loss + neg_loss
        
        return ff_loss
    
    def _compute_goodness(self, x):
        """
        Compute the "goodness" of each layer's representation
        Higher goodness means the representation is more like the training data
        """
        b, t = x.shape
        x = mx.array(x)
        pos = mx.arange(0, t, 1, dtype=x.dtype)
        
        # Use the model's embedding layers
        tok_emb = self.model.wte(x)
        pos_emb = self.model.wpe(pos)
        h = self.model.drop(tok_emb + pos_emb)
        
        # Compute goodness as sum of squared activations across all layers
        goodness = mx.sum(h * h, axis=-1)  # Start with embedding goodness
        
        # Propagate through each layer and accumulate goodness
        for i, block in enumerate(self.model.h):
            h = block.ln_1(h)
            h, _ = block.attn(h)
            goodness = goodness + mx.sum(h * h, axis=-1)
            
            h = block.ln_2(h)
            h = block.mlp(h)
            goodness = goodness + mx.sum(h * h, axis=-1)
        
        # Average across sequence dimension
        goodness = mx.mean(goodness, axis=-1)
        return goodness


def get_memory_usage_gb():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert bytes to GB

class AdaptiveTrainer:
    def __init__(
        self,
        data_path,
        train_config: TrainConfig,
        model_config: GPTConfig,
        checkpoint_dir=None,
    ):
        # Initialize with memory-efficient data loading
        self.data_loader = DataLoader(data_path, model_config.block_size, memory_efficient=True)

        # Base training configuration
        self.train_config = train_config
        self.num_iters = train_config.num_iters
        self.max_lr = train_config.max_lr
        self.min_lr = train_config.min_lr
        self.warmup_iters = train_config.warmup_iters
        self.lr_decay_iters = train_config.lr_decay_iters
        self.rl_update_freq = train_config.rl_update_freq
        
        # Memory management
        self.memory_check_freq = 5  # Check memory usage every N iterations
        self.last_memory_usage = 0
        
        # Dynamic hyperparameters managed by RL
        self.hyperparams = HyperParams(
            learning_rate=train_config.max_lr,
            batch_size=1,
            grad_accumulation_steps=16,
            dropout=model_config.dropout,
            weight_decay=0.01
        )

        # initialize the model and optimizer
        self.model_config = model_config
        self.model = GPT(model_config)
        self.optimizer = AdamW(
            learning_rate=self.hyperparams.learning_rate,
            weight_decay=self.hyperparams.weight_decay
        )
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self.model.loss)
        
        # Initialize RL optimizer for hyperparameters
        self.rl_optimizer = RLHyperOptimizer(
            self.hyperparams,
            exploration_factor=train_config.exploration_factor
        )
        
        # Initialize forward-forward trainer
        self.ff_trainer = ForwardForwardTrainer(self.model, self.data_loader)

        # init gradient accumulation state
        self.accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), self.model.parameters()
        )
        self.accumulated_loss = 0.0
        self.iter_num = 0

        self.checkpoint_dir = checkpoint_dir
        self.data_path = data_path
        
        # For tracking progress
        self.last_losses = []
        self.hp_update_history = []

    def save_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_weights_path = os.path.join(checkpoint_dir, "model_weights.npz")
        model_config_path = os.path.join(checkpoint_dir, "model_config.json")
        hyperopt_path = os.path.join(checkpoint_dir, "hyperopt_state.json")

        self.model.save_weights(model_weights_path)

        with open(model_config_path, "w") as f:
            json.dump(self.model_config.__dict__, f)
            
        # Save hyperparameter optimization state
        with open(hyperopt_path, "w") as f:
            hyperopt_state = {
                "current_hyperparams": self.__dict_from_dataclass(self.hyperparams),
                "best_hyperparams": self.__dict_from_dataclass(self.rl_optimizer.best_hyperparams),
                "best_reward": self.rl_optimizer.best_reward,
                "hp_update_history": [
                    {"iteration": hp["iteration"],
                     "hyperparams": self.__dict_from_dataclass(hp["hyperparams"]),
                     "loss": hp["loss"]}
                    for hp in self.hp_update_history
                ]
            }
            json.dump(hyperopt_state, f, indent=2)
            
    def __dict_from_dataclass(self, dataclass_obj):
        return {field: getattr(dataclass_obj, field) for field in dataclass_obj.__dataclass_fields__}

    def print_parameter_count(self):
        mx.eval(self.model.parameters())
        nparams = sum(x.size for k, x in tree_flatten(self.model.parameters()))
        print(f"Training a custom GPT model with {nparams / 1e6:.3f} M parameters")

    def print_loss(self, iteration_count, average_loss, tic):
        toc = time.perf_counter()
        print(
            f"iter {iteration_count}: train loss {average_loss:.3f}, "
            f"it/sec {1.0 / (toc - tic):.3f}, "
            f"lr {self.optimizer.learning_rate:.4f}, "
            f"bs {self.hyperparams.batch_size}, "
            f"grad_accum {self.hyperparams.grad_accumulation_steps}"
        )
        return toc

    def update_learning_rate(self, it):
        """Default learning rate scheduler with cosine decay"""
        if it < self.warmup_iters:
            return self.hyperparams.learning_rate * it / self.warmup_iters
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        new_lr = self.min_lr + coeff * (self.hyperparams.learning_rate - self.min_lr)

        self.optimizer.set_learning_rate(new_lr)

    def compute_minibatch_loss_grads(self, inputs, targets):
        """Compute loss and gradients for a single minibatch"""
        inputs, targets = map(mx.array, (inputs, targets))
        loss, grads = self.loss_and_grad_fn(inputs, targets)

        self.accumulated_grads = tree_map(
            lambda acc, new: acc + new * (1.0 / self.hyperparams.grad_accumulation_steps),
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
        """Compute average loss over accumulated batches"""
        average_loss = self.accumulated_loss / self.hyperparams.grad_accumulation_steps
        self.accumulated_loss = 0.0
        
        # Use helper function for backwards compatibility
        simplify_compatible(loss, self.model.parameters())
        mx.eval(loss, self.model.parameters())
        
        # Store loss for RL optimizer
        self.last_losses.append(average_loss)
        if len(self.last_losses) > 10:
            self.last_losses.pop(0)
            
        # Update RL optimizer with current loss
        self.rl_optimizer.update_loss(average_loss)
        
        return average_loss

    def perform_gradient_step(self):
        """Perform optimization step with accumulated gradients"""
        self.model.update(
            self.optimizer.apply_gradients(self.accumulated_grads, self.model)
        )
        self.accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), self.model.parameters()
        )
        
    def update_hyperparameters(self):
        """Use RL to update hyperparameters"""
        new_hyperparams = self.rl_optimizer.update()
        
        # Update optimizer with new hyperparameters
        if new_hyperparams.learning_rate != self.hyperparams.learning_rate:
            self.optimizer.set_learning_rate(new_hyperparams.learning_rate)
            
        if new_hyperparams.weight_decay != self.hyperparams.weight_decay:
            self.optimizer.weight_decay = new_hyperparams.weight_decay
        
        # Update model with new dropout
        if new_hyperparams.dropout != self.hyperparams.dropout:
            for block in self.model.h:
                block.attn.attn_dropout.p = new_hyperparams.dropout
                block.attn.resid_dropout.p = new_hyperparams.dropout
                block.mlp.dropout.p = new_hyperparams.dropout
            self.model.drop.p = new_hyperparams.dropout
            
        # Store the updated hyperparameters
        self.hyperparams = new_hyperparams
        
        # Record hyperparameter update
        self.hp_update_history.append({
            "iteration": self.iter_num,
            "hyperparams": self.hyperparams,
            "loss": np.mean(self.last_losses) if self.last_losses else float('inf')
        })
        
        # Print current hyperparameters
        print(f"Updated hyperparameters: lr={self.hyperparams.learning_rate:.6f}, "
              f"bs={self.hyperparams.batch_size}, "
              f"grad_accum={self.hyperparams.grad_accumulation_steps}, "
              f"dropout={self.hyperparams.dropout:.3f}, "
              f"weight_decay={self.hyperparams.weight_decay:.4f}")

    def perform_forward_forward_update(self, inputs):
        """Perform an unsupervised forward-forward update"""
        print("Performing forward-forward unsupervised update...")
        
        # Get forward-forward loss
        ff_loss = self.ff_trainer.forward_forward_step(inputs)
        
        # Get gradients from forward-forward loss
        ff_grad_fn = nn.grad(self.ff_trainer.forward_forward_step)
        ff_grads = ff_grad_fn(inputs)
        
        # Apply gradients
        self.model.update(
            self.optimizer.apply_gradients(ff_grads, self.model)
        )
        
        print(f"Forward-forward loss: {ff_loss.item():.3f}")
        
    def monitor_and_adjust_memory(self):
        """Monitor memory usage and adjust parameters if needed"""
        current_memory = get_memory_usage_gb()
        memory_change = current_memory - self.last_memory_usage
        self.last_memory_usage = current_memory
        
        print(f"Current memory usage: {current_memory:.2f} GB / {MAX_MEMORY_GB} GB")
        
        # If we're approaching the memory limit, adjust parameters
        if current_memory > MAX_MEMORY_GB * 0.85:
            print("WARNING: Memory usage is high, adjusting parameters...")
            
            # First try reducing batch size
            if self.hyperparams.batch_size > 1:
                self.hyperparams.batch_size = max(1, self.hyperparams.batch_size - 1)
                print(f"Reduced batch size to {self.hyperparams.batch_size}")
                return True
                
            # Then try increasing gradient accumulation steps
            elif self.hyperparams.grad_accumulation_steps < 64:
                self.hyperparams.grad_accumulation_steps += 4
                print(f"Increased grad accumulation steps to {self.hyperparams.grad_accumulation_steps}")
                return True
                
            # Last resort: force garbage collection
            else:
                import gc
                gc.collect()
                print("Forced garbage collection")
                return False
                
        return False
    
    def train(self):
        self.print_parameter_count()

        tic = time.perf_counter()
        
        # Check initial memory usage
        self.last_memory_usage = get_memory_usage_gb()
        print(f"Initial memory usage: {self.last_memory_usage:.2f} GB / {MAX_MEMORY_GB} GB")
        
        # Main training loop
        for iteration in range(self.num_iters * self.hyperparams.grad_accumulation_steps):
            # Get batch with current batch size
            train_data = self.data_loader.get_batch_iterator(self.hyperparams.batch_size)
            inputs, targets = next(train_data)
            
            # Standard supervised training step
            loss = self.compute_minibatch_loss_grads(inputs, targets)

            if (iteration + 1) % self.hyperparams.grad_accumulation_steps == 0:
                # Perform gradient update
                self.perform_gradient_step()
                self.update_learning_rate(self.iter_num)
                batch_loss = self.compute_batch_loss(loss)
                tic = self.print_loss(self.iter_num, batch_loss, tic)
                
                # Check memory usage periodically
                if self.iter_num > 0 and self.iter_num % self.memory_check_freq == 0:
                    self.monitor_and_adjust_memory()
                
                # Periodically update hyperparameters using RL
                if self.iter_num > 0 and self.iter_num % self.rl_update_freq == 0:
                    self.update_hyperparameters()
                
                # Periodically use forward-forward algorithm
                if self.iter_num > 0 and self.iter_num % self.ff_trainer.ff_update_freq == 0:
                    # Check memory before expensive operation
                    current_memory = get_memory_usage_gb()
                    if current_memory < MAX_MEMORY_GB * 0.8:  # Only run if memory is available
                        self.perform_forward_forward_update(inputs)
                    else:
                        print("Skipping forward-forward update due to high memory usage")

                if self.iter_num % self.train_config.save_every == 0:
                    print(f"Saving model to {self.checkpoint_dir}")
                    self.save_checkpoint(self.checkpoint_dir)

                self.iter_num += 1


def adjust_model_size_for_memory(init_memory_gb):
    """Adjust model size based on available memory"""
    available_memory = MAX_MEMORY_GB - init_memory_gb
    
    # Default large configuration (GPT-2 XL size)
    n_layer = 48
    n_head = 25
    n_embd = 1600
    
    # Scale down if less than 15GB available
    if available_memory < 15:
        scale_factor = available_memory / 15
        n_layer = max(8, int(n_layer * scale_factor))
        n_head = max(8, int(n_head * scale_factor))
        n_embd = max(768, int(n_embd * scale_factor))
        
        print(f"Scaled down model for memory constraints:")
        print(f"  - Layers: {n_layer} (from 48)")
        print(f"  - Heads: {n_head} (from 25)")
        print(f"  - Embedding size: {n_embd} (from 1600)")
    
    return n_layer, n_head, n_embd

def main(train_path, checkpoint_dir):
    # Check initial memory usage before loading model
    init_memory = get_memory_usage_gb()
    print(f"Initial memory usage before model: {init_memory:.2f} GB / {MAX_MEMORY_GB} GB")
    
    # Adjust model size based on available memory
    n_layer, n_head, n_embd = adjust_model_size_for_memory(init_memory)
    
    # Configure model with potentially adjusted size
    model_config = GPTConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=50304,
        block_size=1024,
        bias=True,
        dropout=0.1  # Start with some dropout
    )

    train_config = TrainConfig(
        num_iters=200,
        max_lr=1e-4,
        min_lr=1e-5,
        warmup_iters=20,
        lr_decay_iters=200,
        save_every=10,
        rl_update_freq=5,  # Update hyperparameters every 5 iterations
        exploration_factor=0.3  # 30% exploration vs 70% exploitation
    )
    
    trainer = AdaptiveTrainer(train_path, train_config, model_config, checkpoint_dir)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 model with adaptive hyperparameters and memory constraints"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="train.npy",
        help="Path to training data *.npy file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints_adaptive",
        help="Path to checkpoint directory to save model weights",
    )
    parser.add_argument(
        "--max_memory_gb",
        type=float,
        default=MAX_MEMORY_GB,
        help="Maximum memory usage in GB (default: 20)",
    )
    args = parser.parse_args()

    # Update memory constraint if specified via command line
    if args.max_memory_gb != MAX_MEMORY_GB:

        MAX_MEMORY_GB = args.max_memory_gb
        print(f"Memory usage constrained to {MAX_MEMORY_GB} GB")

    main(args.data_path, args.checkpoint_dir)
