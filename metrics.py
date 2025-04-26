"""
Metrics and monitoring utilities for GPT training.
Provides functions for tracking and visualizing model performance.
"""

import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Union, Tuple

# Optional imports for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class MetricsTracker:
    """Track and visualize training metrics."""
    
    def __init__(
        self, 
        log_dir: str, 
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "gpt2-mlx",
        wandb_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory to save logs and visualizations
            use_tensorboard: Whether to use TensorBoard for logging
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
            wandb_name: W&B run name (defaults to timestamp)
            config: Configuration dictionary to log
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = defaultdict(list)
        self.timestamps = defaultdict(list)
        self.start_time = time.time()
        
        # TensorBoard setup
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
        else:
            self.tb_writer = None
            
        # Weights & Biases setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=config,
                dir=log_dir
            )
    
    def add_scalar(
        self, 
        name: str, 
        value: float, 
        step: int,
        group: Optional[str] = None
    ):
        """
        Add a scalar metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
            group: Optional group name for organizing metrics
        """
        if group:
            key = f"{group}/{name}"
        else:
            key = name
            
        self.metrics[key].append(value)
        self.timestamps[key].append(step)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_scalar(key, value, step)
            
        # Log to W&B
        if self.use_wandb:
            wandb.log({key: value}, step=step)
    
    def add_scalars(
        self, 
        metrics_dict: Dict[str, float], 
        step: int,
        group: Optional[str] = None
    ):
        """
        Add multiple scalar metrics at once.
        
        Args:
            metrics_dict: Dictionary of metric names to values
            step: Training step
            group: Optional group name for organizing metrics
        """
        for name, value in metrics_dict.items():
            self.add_scalar(name, value, step, group)
    
    def add_text(self, name: str, text: str, step: int):
        """
        Log text data (e.g., generated samples).
        
        Args:
            name: Name for the text
            text: Text content
            step: Training step
        """
        if self.use_tensorboard:
            self.tb_writer.add_text(name, text, step)
            
        if self.use_wandb:
            wandb.log({name: wandb.Html(f"<pre>{text}</pre>")}, step=step)
    
    def add_generated_samples(
        self, 
        prompts: List[str], 
        generations: List[str], 
        step: int
    ):
        """
        Log model-generated text samples.
        
        Args:
            prompts: List of prompt texts
            generations: List of generated continuations
            step: Training step
        """
        sample_text = ""
        for i, (prompt, gen) in enumerate(zip(prompts, generations)):
            sample_text += f"Sample {i+1}:\nPrompt: {prompt}\nGeneration: {gen}\n\n"
            
        self.add_text("generated_samples", sample_text, step)
    
    def plot_loss_curves(
        self, 
        output_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot training and validation loss curves.
        
        Args:
            output_path: Path to save the figure (default: log_dir/loss_curves.png)
            show: Whether to display the plot
            
        Returns:
            The matplotlib figure
        """
        if output_path is None:
            output_path = os.path.join(self.log_dir, "loss_curves.png")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training loss
        if "train/loss" in self.metrics:
            ax.plot(
                self.timestamps["train/loss"], 
                self.metrics["train/loss"], 
                label="Training Loss"
            )
            
        # Plot validation loss
        if "val/loss" in self.metrics:
            ax.plot(
                self.timestamps["val/loss"], 
                self.metrics["val/loss"], 
                label="Validation Loss"
            )
            
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def save_metrics(self, filepath: Optional[str] = None):
        """
        Save metrics to a JSON file.
        
        Args:
            filepath: Path to save metrics (default: log_dir/metrics.json)
        """
        if filepath is None:
            filepath = os.path.join(self.log_dir, "metrics.json")
            
        # Convert defaultdict to regular dict for JSON serialization
        metrics_dict = {
            "metrics": dict(self.metrics),
            "timestamps": dict(self.timestamps),
            "duration": time.time() - self.start_time
        }
        
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f, indent=2)
    
    def close(self):
        """Cleanup resources."""
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()
            
        if self.use_wandb:
            wandb.finish()


class PerplexityCalculator:
    """Calculate perplexity metrics for language models."""
    
    @staticmethod
    def calculate_perplexity(loss: float) -> float:
        """
        Calculate perplexity from cross-entropy loss.
        
        Args:
            loss: Cross-entropy loss value
            
        Returns:
            Perplexity value
        """
        return np.exp(loss)
    
    @staticmethod
    def calculate_bpc(loss: float) -> float:
        """
        Calculate bits per character from loss.
        
        Args:
            loss: Cross-entropy loss value
            
        Returns:
            Bits per character value
        """
        return loss / np.log(2)


class LearningCurveAnalyzer:
    """
    Analyze learning curves to provide insights into training dynamics.
    """
    
    @staticmethod
    def detect_overfitting(
        train_losses: List[float],
        val_losses: List[float],
        window_size: int = 5
    ) -> Tuple[bool, float]:
        """
        Detect signs of overfitting by comparing smoothed train and val losses.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            window_size: Window size for smoothing
            
        Returns:
            (is_overfitting, divergence_score)
        """
        if len(train_losses) != len(val_losses) or len(train_losses) < window_size*2:
            return False, 0.0
        
        # Apply simple moving average smoothing
        def smooth(losses, window):
            return [
                np.mean(losses[max(0, i-window):i+1]) 
                for i in range(len(losses))
            ]
        
        smoothed_train = smooth(train_losses, window_size)
        smoothed_val = smooth(val_losses, window_size)
        
        # Check if validation loss is increasing while training loss is decreasing
        train_trend = smoothed_train[-window_size] - smoothed_train[-1]
        val_trend = smoothed_val[-1] - smoothed_val[-window_size]
        
        # Divergence score: positive if overfitting suspected
        divergence_score = train_trend + val_trend
        is_overfitting = train_trend > 0 and val_trend > 0 and divergence_score > 0
        
        return is_overfitting, divergence_score
    
    @staticmethod
    def suggest_learning_rate(
        losses: List[float],
        learning_rates: List[float]
    ) -> Optional[float]:
        """
        Suggest a learning rate based on loss curve during LR finder.
        
        Args:
            losses: List of loss values
            learning_rates: List of learning rates used
            
        Returns:
            Suggested learning rate or None if can't determine
        """
        if len(losses) < 10 or len(losses) != len(learning_rates):
            return None
        
        # Calculate smoothed derivatives
        derivatives = []
        window = min(5, len(losses) // 5)
        
        for i in range(window, len(losses)-window):
            derivative = (losses[i+window] - losses[i-window]) / (
                np.log10(learning_rates[i+window]) - np.log10(learning_rates[i-window])
            )
            derivatives.append((learning_rates[i], derivative))
        
        # Find steepest negative slope
        min_derivative = float('inf')
        best_lr = None
        
        for lr, derivative in derivatives:
            if derivative < min_derivative:
                min_derivative = derivative
                best_lr = lr
        
        # Use a slightly lower learning rate than steepest point
        if best_lr is not None:
            return best_lr / 10.0
        
        return None