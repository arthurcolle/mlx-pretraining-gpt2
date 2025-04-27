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
import psutil

from mlx.utils import tree_flatten, tree_map
from transformer import GPT, GPTConfig
from dataclasses import dataclass
from metrics import MetricsTracker, PerplexityCalculator, LearningCurveAnalyzer
from sample_generator import TextSampleGenerator
from helper_function import simplify_compatible
from train import DataLoader, TrainConfig, GPTTrainer


class MemoryEfficientDataLoader:
    """Memory-efficient data loader that streams data instead of preloading."""
    def __init__(self, data_path, context_size):
        self.data_path = data_path
        self.context_size = context_size
        
        # Use memory-mapped file for efficient access
        self.dataset = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.dataset)
        self.window_size = self.context_size + 1
        self.examples = self.num_tokens - self.window_size + 1
        
        print(f"Using memory-efficient data loading with {self.num_tokens} tokens")
        print(f"Context size: {context_size}, Potential training examples: {self.examples}")
    
    def get_random_batch(self, batch_size):
        """Get a random batch of data without preloading everything."""
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
        """Return an iterator that yields random batches."""
        while True:
            yield self.get_random_batch(batch_size)


class SmallGPTTrainer(GPTTrainer):
    """Enhanced trainer for small GPT models with memory efficiency and better sampling."""
    
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
        memory_efficient=True,
    ):
        # Initialize parent class
        super().__init__(
            data_path, 
            val_path, 
            train_config, 
            model_config, 
            checkpoint_dir,
            use_wandb,
            use_tensorboard,
            sample_prompts
        )
        
        # Replace standard data loader with memory-efficient version if requested
        if memory_efficient:
            self.data_loader = MemoryEfficientDataLoader(data_path, model_config.block_size)
            if val_path:
                self.val_loader = MemoryEfficientDataLoader(val_path, model_config.block_size)
            print("Using memory-efficient data loading")
        
        # Advanced generation parameters
        self.temperature = 0.8  # Slightly higher temperature for more creative generations
        self.top_k = 40
        self.top_p = 0.9
        
        # Memory monitoring
        self.memory_check_freq = 5  # Check memory every 5 iterations
        self.last_memory_usage = self._get_memory_usage()
        
        # Learning curve analysis
        self.curve_analyzer = LearningCurveAnalyzer()
        
    def _get_memory_usage(self):
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 ** 3)  # Convert bytes to GB
    
    def monitor_memory(self):
        """Monitor memory usage during training"""
        current_memory = self._get_memory_usage()
        memory_change = current_memory - self.last_memory_usage
        self.last_memory_usage = current_memory
        
        # Log memory usage
        self.metrics.add_scalar("memory_usage_gb", current_memory, self.iter_num, group="system")
        
        print(f"Memory usage: {current_memory:.2f} GB (Δ {memory_change:+.2f} GB)")
        return current_memory
    
    def generate_samples(self):
        """Generate text samples using enhanced sampling parameters"""
        try:
            import tiktoken
            # Load tokenizer
            tokenizer = tiktoken.get_encoding("gpt2")
            
            # Prepare prompts (limit to num_samples)
            prompts = self.sample_prompts[:self.num_samples]
            
            # Create sample generator with enhanced parameters
            generator = TextSampleGenerator(
                model=self.model,
                prompts=prompts,
                max_new_tokens=self.max_sample_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
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
        """Enhanced training loop with memory monitoring and learning curve analysis"""
        self.print_parameter_count()

        tic = time.perf_counter()
        train_data = self.data_loader.get_batch_iterator(self.batch_size)

        # Initial memory check
        initial_memory = self.monitor_memory()
        print(f"Starting training with initial memory usage: {initial_memory:.2f} GB")

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
                
                # Check memory usage periodically
                if self.iter_num > 0 and self.iter_num % self.memory_check_freq == 0:
                    self.monitor_memory()

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
                        
                        # Check for overfitting
                        if len(self.train_losses) >= 10 and len(self.val_losses) >= 5:
                            is_overfitting, score = self.curve_analyzer.detect_overfitting(
                                self.train_losses[-10:], 
                                self.val_losses[-5:]
                            )
                            if is_overfitting:
                                print(f"Overfitting detected (score: {score:.3f}). Consider early stopping.")
                        
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
                    
                    # Generate learning curve plots
                    self.metrics.plot_loss_curves()

                self.iter_num += 1

        # Final checkpoint and metrics
        self.save_checkpoint(self.checkpoint_dir)
        self.metrics.save_metrics()
        
        # Generate final samples
        self.generate_samples()
        
        # Final memory check
        final_memory = self.monitor_memory()
        print(f"Training complete. Final memory usage: {final_memory:.2f} GB")


def main(train_path, val_path, checkpoint_dir, use_wandb=False, use_tensorboard=True, memory_efficient=True):
    # Small model config (6-layer model with fewer parameters)
    model_config = GPTConfig(
        n_layer=6,
        n_head=8,
        n_embd=512,
        vocab_size=50304,
        block_size=512,
        bias=True,
        dropout=0.1,  # Add dropout for better generalization
    )

    train_config = TrainConfig(
        num_iters=100,
        batch_size=2,
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
        "The future of AI is",
        "She looked at him and said",
        "The most important thing to remember"
    ]
    
    trainer = SmallGPTTrainer(
        train_path, 
        val_path, 
        train_config, 
        model_config, 
        checkpoint_dir,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        sample_prompts=sample_prompts,
        memory_efficient=memory_efficient
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
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        default=True,
        help="Whether to use memory-efficient data loading",
    )
    args = parser.parse_args()

    main(args.data_path, args.val_path, args.checkpoint_dir, 
         use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard,
         memory_efficient=args.memory_efficient)