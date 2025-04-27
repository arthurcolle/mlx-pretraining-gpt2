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
    sample_every: int
    num_samples: int
    max_sample_tokens: int


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
        use_wandb=False,
        use_tensorboard=True,
        sample_prompts=None,
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
        self.sample_every = train_config.sample_every
        self.num_samples = train_config.num_samples
        self.max_sample_tokens = train_config.max_sample_tokens
        self.sample_prompts = sample_prompts or ["Once upon a time", "The meaning of life is", "In the future"]

        # initialize the model and optimizer
        self.model_config = model_config
        self.model = GPT(model_config)
        
        # Check for multi-device training
        self.devices = mx.available_devices()
        if len(self.devices) > 1 and hasattr(mx, 'distribute'):
            from mlx.distribute import distribute
            print(f"Distributing model across {len(self.devices)} devices")
            self.model = distribute(self.model, devices=self.devices)
            
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

        # Initialize directories
        self.samples_dir = os.path.join(checkpoint_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # Setup metrics tracker
        self.metrics = MetricsTracker(
            log_dir=checkpoint_dir,
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb,
            wandb_project="gpt2-mlx",
            wandb_name=f"gpt2-{model_config.n_layer}l-{model_config.n_head}h-{model_config.n_embd}d",
            config={
                "model": model_config.__dict__,
                "train": train_config.__dict__,
                "dataset": {
                    "train_path": data_path,
                    "val_path": val_path,
                }
            }
        )

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
            f"ppl {PerplexityCalculator.calculate_perplexity(average_loss):.2f}, "
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
        
        # Convert inputs and targets to float16 for mixed precision training
        if hasattr(self.model_config, 'use_fp16') and self.model_config.use_fp16:
            inputs = inputs.astype(mx.float16)
        
        # Use JIT compilation for faster execution
        @mx.compile
        def _loss_and_grad_fn(inputs, targets):
            return self.loss_and_grad_fn(inputs, targets)
            
        loss, grads = _loss_and_grad_fn(inputs, targets)

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
        # simplify_compatible is used instead of mx.simplify which was removed in newer MLX versions
        simplify_compatible(loss, self.model.parameters())
        mx.eval(loss, self.model.parameters())
        return average_loss

    def perform_gradient_step(self):
        self.model.update(
            self.optimizer.apply_gradients(self.accumulated_grads, self.model)
        )
        self.accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), self.model.parameters()
        )

    def evaluate(self, num_batches=10):
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

    def generate_samples(self):
        """Generate text samples to track qualitative model progress"""
        try:
            import tiktoken
            # Load tokenizer
            tokenizer = tiktoken.get_encoding("gpt2")
            
            # Prepare prompts (limit to num_samples)
            prompts = self.sample_prompts[:self.num_samples]
            
            # Create sample generator
            generator = TextSampleGenerator(
                model=self.model,
                prompts=prompts,
                max_new_tokens=self.max_sample_tokens,
                temperature=0.7,
                top_k=40,
                top_p=0.9
            )
            
            # Generate samples
            print(f"Generating {len(prompts)} text samples...")
            samples = generator.generate_samples()
            
            if not samples:
                print("WARNING: No samples were generated")
                return
                
            # Save samples
            output_path = os.path.join(self.samples_dir, f"samples_iter_{self.iter_num}.json")
            generator.save_samples(samples, output_path)
            
            # Log samples to trackers
            prompts_text = []
            generations_text = []
            for prompt, generated, _ in samples:
                prompts_text.append(prompt)
                generations_text.append(generated)
            
            self.metrics.add_generated_samples(prompts_text, generations_text, self.iter_num)
            
            # Log a sample to console for quick inspection
            if samples:
                prompt, generated, full = samples[0]
                print(f"Sample generation (first 100 chars): {full[:100]}...")
            
            print(f"Generated {len(samples)} text samples. Saved to {output_path}")
            
        except ImportError as e:
            print(f"Sample generation skipped: {str(e)}")
        except Exception as e:
            import traceback
            print(f"Error generating samples: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            # Continue training despite generation error
            print("Continuing training despite sample generation error...")

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
                
                # Log metrics
                perplexity = PerplexityCalculator.calculate_perplexity(batch_loss)
                lr = self.optimizer.learning_rate
                self.metrics.add_scalars(
                    {
                        "loss": batch_loss,
                        "perplexity": perplexity,
                        "learning_rate": lr
                    },
                    step=self.iter_num,
                    group="train"
                )
                
                tic = self.print_loss(self.iter_num, batch_loss, tic)

                # Validation
                if self.val_loader and self.iter_num % self.validation_interval == 0:
                    val_loss = self.evaluate()
                    self.val_losses.append(val_loss)
                    val_ppl = PerplexityCalculator.calculate_perplexity(val_loss)
                    
                    # Log validation metrics
                    self.metrics.add_scalars(
                        {
                            "loss": val_loss,
                            "perplexity": val_ppl
                        },
                        step=self.iter_num,
                        group="val"
                    )
                    
                    print(f"iter {self.iter_num}: val loss {val_loss:.3f}, val ppl {val_ppl:.2f}")
                    
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
                            # Save final metrics
                            self.metrics.save_metrics()
                            # Generate final samples
                            self.generate_samples()
                            return

                # Generate text samples
                if self.iter_num % self.sample_every == 0:
                    self.generate_samples()

                # Checkpoint saving
                if self.iter_num % self.train_config.save_every == 0:
                    print(f"Saving model to {self.checkpoint_dir}")
                    self.save_checkpoint(self.checkpoint_dir)
                    self.metrics.save_metrics()

                self.iter_num += 1

        # Final checkpoint and metrics
        self.save_checkpoint(self.checkpoint_dir)
        self.metrics.save_metrics()
        
        # Generate final samples
        self.generate_samples()


def main(train_path, val_path, checkpoint_dir, use_wandb=False, use_tensorboard=True, use_fp16=False, distributed=False):
    model_config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        vocab_size=50304,
        block_size=1024,
        bias=True,
        use_fp16=use_fp16,
    )

    train_config = TrainConfig(
        num_iters=200,
        batch_size=2,
        grad_accumulation_steps=4,
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_iters=20,
        lr_decay_iters=200,
        save_every=20,
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
        "The future of AI is",
        "She looked at him and said",
        "The most important thing to remember"
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
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GPT-2-style model on a custom dataset"
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
        default="checkpoints",
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
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Enable mixed precision training with float16",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable multi-device distributed training",
    )
    args = parser.parse_args()

    main(args.data_path, args.val_path, args.checkpoint_dir, 
         use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard,
         use_fp16=args.use_fp16)
