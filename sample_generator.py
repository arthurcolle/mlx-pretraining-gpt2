"""
Text sample generator for GPT-MLX models.
Generates text samples during training to evaluate qualitative model performance.
"""

import argparse
import tiktoken
import numpy as np
import mlx.core as mx
import os
import json
import time
from typing import List, Dict, Optional, Union, Tuple

from transformer import GPT, GPTConfig


class TextSampleGenerator:
    """Generate text samples from a trained GPT model."""
    
    def __init__(
        self,
        model: GPT,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        tokenizer_name: str = "gpt2"
    ):
        """
        Initialize text sample generator.
        
        Args:
            model: Trained GPT model
            prompts: List of text prompts for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (higher = more random)
            top_k: Number of highest probability tokens to consider (None for all)
            top_p: Cumulative probability threshold for nucleus sampling (None to disable)
            tokenizer_name: Tokenizer name for encoding/decoding text
        """
        self.model = model
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    def _encode_prompt(self, prompt: str) -> mx.array:
        """
        Encode a text prompt to token IDs.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Token IDs as an MLX array
        """
        tokens = self.tokenizer.encode(prompt)
        return mx.array([tokens], dtype=mx.int32)
    
    def _decode_tokens(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(tokens)
    
    def _sample_with_temperature(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> mx.array:
        """
        Sample from logits with temperature, optional top-k, and optional nucleus sampling.
        
        Args:
            logits: Model logits
            temperature: Temperature for sampling
            top_k: Number of top tokens to consider
            top_p: Probability threshold for nucleus sampling
            
        Returns:
            Sampled token ID
        """
        # Ensure logits have correct shape (batch_size, vocab_size)
        if logits.ndim > 2:
            print(f"Warning: Unexpected logits shape {logits.shape}, reshaping")
            logits = logits.reshape(logits.shape[0], -1)
            
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # For temperature=0, use greedy sampling
            return mx.argmax(logits, axis=-1)
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            v = mx.topk(logits, min(top_k, logits.shape[-1]))
            # Create a mask for the top-k logits
            min_value = v[:, -1].reshape(-1, 1)
            logits = mx.where(logits < min_value, float('-inf'), logits)
        
        # Apply nucleus (top-p) sampling
        if top_p is not None and 0.0 < top_p < 1.0:
            # Sort logits in descending order - MLX sort doesn't have descending param
            sorted_logits = mx.sort(logits, axis=-1)
            sorted_logits = sorted_logits[:, ::-1]  # Reverse to get descending order
            # Get indices with argsort and reverse
            sorted_indices = mx.argsort(logits, axis=-1)
            sorted_indices = sorted_indices[:, ::-1]  # Reverse to match descending order
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
            
            # Create a mask for tokens with cumulative probability < top_p
            sorted_mask = cumulative_probs <= top_p
            
            # Add the first token which exceeds top_p to the mask
            # Shift the mask by 1 to the right and set the first position to True
            shifted_mask = mx.concatenate([
                mx.ones_like(sorted_mask[:, :1]), 
                sorted_mask[:, :-1]
            ], axis=-1)
            sorted_mask = mx.logical_or(sorted_mask, shifted_mask)
            
            # Apply top-p filtering directly by modifying logits
            for i in range(logits.shape[0]):
                # Create a new temporary logits array with -inf values
                logits_masked = mx.full(logits[i].shape, float('-inf'))
                # Only keep the values within the mask
                for j in range(sorted_indices[i].size):
                    if j < sorted_mask[i].size and sorted_mask[i][j]:
                        idx = sorted_indices[i][j]
                        # In MLX 0.20.0, use direct assignment
                        logits_masked[idx] = logits[i, idx]
                # Update the entire row for this batch
                # In MLX 0.20.0, use direct assignment instead of at[].set()
                logits[i] = logits_masked
        
        # Sample from the filtered distribution
        try:
            return mx.random.categorical(logits)
        except Exception as e:
            print(f"Error sampling from logits: {e}")
            print(f"Logits shape: {logits.shape}, min: {mx.min(logits)}, max: {mx.max(logits)}")
            # Fall back to greedy sampling in case of error
            print("Falling back to greedy sampling")
            return mx.argmax(logits, axis=-1)
    
    def generate_samples(self) -> List[Tuple[str, str, str]]:
        """
        Generate text samples for all prompts.
        
        Returns:
            List of tuples (prompt, generated_text, full_text)
        """
        results = []
        
        for prompt in self.prompts:
            try:
                print(f"Generating from prompt: {prompt[:50]}...")
                
                # Encode prompt
                input_ids = self._encode_prompt(prompt)
                input_pos = mx.arange(0, input_ids.shape[1], dtype=mx.int32)
                
                # Initialize generation state
                generated_ids = []
                
                # Create attention mask (causal for self-attention)
                mask = self.model._create_causal_mask(input_ids.shape[1])
                
                # Initial forward pass to get kv cache
                logits, kv_cache = self.model._forward_transformer_blocks(
                    input_ids, input_pos, mask=mask, build_cache=True
                )
                
                # Get the logits for the last token
                next_logits = (logits[:, -1, :] @ self.model.wte.weight.T)
                # Make sure the shape is correct for sampling
                if next_logits.ndim > 2:
                    next_logits = next_logits[:, -1, :]
                
                # Sample next token
                next_token = self._sample_with_temperature(
                    next_logits, self.temperature, self.top_k, self.top_p
                )
                generated_ids.append(next_token.item())
                
                # Generate remaining tokens
                for i in range(self.max_new_tokens - 1):
                    # Prepare input for next token (just the last generated token)
                    next_input = mx.array([[next_token.item()]], dtype=mx.int32)
                    next_pos = mx.array([[input_ids.shape[1] + i]], dtype=mx.int32)
                    
                    # Forward pass with cached key/values
                    logits, kv_cache = self.model._forward_transformer_blocks(
                        next_input, next_pos, cache=kv_cache
                    )
                    
                    # Get next token logits - make sure to use the right shape
                    next_logits = (logits @ self.model.wte.weight.T)
                    # Only consider the last position (though with single token input, this is redundant)
                    if next_logits.ndim > 2:
                        next_logits = next_logits[:, -1, :]
                    
                    # Sample next token
                    next_token = self._sample_with_temperature(
                        next_logits, self.temperature, self.top_k, self.top_p
                    )
                    generated_ids.append(next_token.item())
                
                # Decode generated text
                generated_text = self._decode_tokens(generated_ids)
                full_text = prompt + generated_text
                
                results.append((prompt, generated_text, full_text))
                
                # Print a short preview of the generated text
                print(f"Generated: {generated_text[:50]}..." if len(generated_text) > 50 else f"Generated: {generated_text}")
                
            except Exception as e:
                import traceback
                print(f"Error generating sample for prompt '{prompt[:30]}...': {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                # Continue with next prompt
                print("Skipping this prompt and continuing with others...")
        
        return results
    
    def save_samples(self, samples: List[Tuple[str, str, str]], output_path: str):
        """
        Save generated samples to a file.
        
        Args:
            samples: List of generated samples (prompt, generated_text, full_text)
            output_path: Path to save samples
        """
        samples_data = []
        
        for i, (prompt, generated, full) in enumerate(samples):
            samples_data.append({
                "sample_id": i,
                "prompt": prompt,
                "generated": generated,
                "full_text": full,
                "generation_config": {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p
                }
            })
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(samples_data, f, indent=2)


def load_model_from_checkpoint(checkpoint_dir: str) -> GPT:
    """
    Load a GPT model from checkpoint.
    
    Args:
        checkpoint_dir: Directory containing model weights and config
        
    Returns:
        Loaded GPT model
    """
    # Load model configuration
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Create model config
    model_config = GPTConfig(**config_dict)
    
    # Create model
    model = GPT(model_config)
    
    # Try to load best model if it exists, otherwise load regular model
    best_weights_path = os.path.join(checkpoint_dir, "best_model_weights.npz")
    regular_weights_path = os.path.join(checkpoint_dir, "model_weights.npz")
    
    if os.path.exists(best_weights_path):
        print(f"Loading best model weights from {best_weights_path}")
        weights_path = best_weights_path
    elif os.path.exists(regular_weights_path):
        print(f"Loading model weights from {regular_weights_path}")
        weights_path = regular_weights_path
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
    
    # Load weights
    model.load_weights(weights_path)
    
    return model


def main():
    """Run the sample generator from command line."""
    parser = argparse.ArgumentParser(
        description="Generate text samples from a trained GPT-MLX model"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing model weights and config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples",
        help="Directory to save generated samples",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="JSON file containing prompts (list of strings)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt for generation (if prompt_file not provided)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (higher = more random)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Number of highest probability tokens to consider (0 to disable)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Cumulative probability threshold for nucleus sampling (0 to disable)",
    )
    
    args = parser.parse_args()
    
    # Process top_k and top_p (None if disabled)
    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p > 0 else None
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint_dir)
    
    # Load prompts
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = json.load(f)
    else:
        prompts = [args.prompt]
    
    # Create generator
    generator = TextSampleGenerator(
        model=model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    # Generate samples
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    samples = generator.generate_samples()
    
    # Save samples
    output_path = os.path.join(args.output_dir, f"samples_{timestamp}.json")
    generator.save_samples(samples, output_path)
    
    print(f"\nGeneration complete.")
    print(f"Generated {len(samples)} samples. Results saved to {output_path}")
    
    # Print a sample
    print("\nSample output:")
    prompt, generated, full = samples[0]
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()