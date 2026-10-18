# Author: Rayaan tasnim, 13 aged prodigy

import os
import sys
import math
import logging
import traceback
from typing import Tuple, Optional, Dict, Any

import torch

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TransformerDecoderBlock")


class RMSNorm:
    """
    Custom Root Mean Square Layer Normalization (RMSNorm) implemented natively 
    using raw tensor operations without abstract nn.Module layer definitions.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        try:
            self.eps = eps
            # Trainable scaling parameter (gain)
            self.weight = torch.nn.Parameter(torch.ones(dim))
            logger.info(f"Initialized RMSNorm with dimension {dim} and epsilon {eps}")
        except Exception as e:
            logger.error(f"Failed to initialize RMSNorm: {e}")
            traceback.print_exc()
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMSNorm to input tensor x of shape (..., dim).
        """
        try:
            # RMS computation: sqrt(mean(x^2) + eps)
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            rsigma = torch.rsqrt(variance + self.eps)
            normalized = x * rsigma
            out = normalized * self.weight
            return out
        except Exception as e:
            logger.error(f"Error during RMSNorm forward pass: {e}")
            traceback.print_exc()
            raise


class KVCache:
    """
    Key-Value Cache module for efficient auto-regressive text generation.
    Stores historical keys and values to prevent redundant computations.
    """
    def __init__(self):
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value tensors along the sequence dimension,
        and returns the full concatenated cache.
        """
        try:
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                # Concatenate along the sequence dimension (dim=2 for shape [batch, heads, seq_len, head_dim])
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)
            return self.k_cache, self.v_cache
        except Exception as e:
            logger.error(f"Error updating KV cache: {e}")
            traceback.print_exc()
            raise

    def reset(self):
        """Resets the KV cache state."""
        try:
            self.k_cache = None
            self.v_cache = None
        except Exception as e:
            logger.error(f"Error resetting KV cache: {e}")
            traceback.print_exc()
            raise


class DecoderTransformerLayer:
    """
    Production-grade, single Decoder-only Transformer block implemented from scratch
    using raw tensor computations, custom multi-head attention, KV caching, RMSNorm,
    and a Feed-Forward Network (FFN).
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-5):
        try:
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.d_v = d_model // n_heads
            self.eps = eps

            if d_model % n_heads != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

            # Custom initialization for weights using Xavier / Kaiming style scaling
            scale_qkv = 1.0 / math.sqrt(d_model)
            scale_proj = 1.0 / math.sqrt(d_model)
            scale_ffn1 = 1.0 / math.sqrt(d_model)
            scale_ffn2 = 1.0 / math.sqrt(d_ff)

            # Manual Weight and Bias tensors (Registered as nn.Parameter for parameter tracking if needed)
            self.W_q = torch.nn.Parameter(torch.randn(d_model, d_model) * scale_qkv)
            self.b_q = torch.nn.Parameter(torch.zeros(d_model))

            self.W_k = torch.nn.Parameter(torch.randn(d_model, d_model) * scale_qkv)
            self.b_k = torch.nn.Parameter(torch.zeros(d_model))

            self.W_v = torch.nn.Parameter(torch.randn(d_model, d_model) * scale_qkv)
            self.b_v = torch.nn.Parameter(torch.zeros(d_model))

            self.W_o = torch.nn.Parameter(torch.randn(d_model, d_model) * scale_proj)
            self.b_o = torch.nn.Parameter(torch.zeros(d_model))

            # Feed-Forward Network Weights
            self.W_fc1 = torch.nn.Parameter(torch.randn(d_model, d_ff) * scale_ffn1)
            self.b_fc1 = torch.nn.Parameter(torch.zeros(d_ff))

            self.W_fc2 = torch.nn.Parameter(torch.randn(d_ff, d_model) * scale_ffn2)
            self.b_fc2 = torch.nn.Parameter(torch.zeros(d_model))

            # Normalization Layers
            self.rms_norm_1 = RMSNorm(d_model, eps=eps)
            self.rms_norm_2 = RMSNorm(d_model, eps=eps)

            # KV Cache instance
            self.kv_cache = KVCache()

            logger.info(
                f"DecoderTransformerLayer initialized successfully: "
                f"d_model={d_model}, n_heads={n_heads}, d_k={self.d_k}, d_ff={d_ff}"
            )
        except Exception as e:
            logger.error(f"Initialization failure in DecoderTransformerLayer: {e}")
            traceback.print_exc()
            raise

    def _apply_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Executes scaled dot-product multi-head attention with KV caching support.
        """
        try:
            batch_size, n_heads, seq_len_q, d_k = q.shape
            
            # Scaled dot-product attention scores
            # q: [batch, heads, seq_q, d_k], k: [batch, heads, seq_k, d_k]
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

            if mask is not None:
                # Apply causal mask (e.g., -inf where mask == 0)
                scores = scores.masked_fill(mask == 0, float("-inf"))

            # Softmax normalization across key sequence dimension
            attn_weights = torch.softmax(scores, dim=-1)

            # Compute weighted values
            # attn_weights: [batch, heads, seq_q, seq_k], v: [batch, heads, seq_k, d_v]
            context = torch.matmul(attn_weights, v)
            return context
        except Exception as e:
            logger.error(f"Error during attention matrix computation: {e}")
            traceback.print_exc()
            raise

    def forward(self, x: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """
        Performs a full forward pass through the Decoder-only Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            use_cache: Boolean flag to determine whether to utilize the KV cache.
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        try:
            batch_size, seq_len, _ = x.shape

            # ==========================================
            # 1. Sub-layer 1: Pre-RMSNorm + Multi-Head Attention + Residual Connection
            # ==========================================
            norm_x = self.rms_norm_1.forward(x)

            # Linear projections for Q, K, V
            q_proj = torch.matmul(norm_x, self.W_q) + self.b_q
            k_proj = torch.matmul(norm_x, self.W_k) + self.b_k
            v_proj = torch.matmul(norm_x, self.W_v) + self.b_v

            # Reshape and transpose for multi-head attention:
            # [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
            q = q_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            k = k_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            v = v_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

            # Handle KV Caching
            if use_cache:
                k, v = self.kv_cache.update(k, v)

            seq_len_k = k.shape[2]

            # Construct causal mask
            mask = None
            if seq_len > 1:
                mask = torch.tril(torch.ones(seq_len, seq_len_k, device=x.device)).view(1, 1, seq_len, seq_len_k)
            elif seq_len == 1 and seq_len_k > 1:
                mask = torch.ones(1, 1, 1, seq_len_k, device=x.device)

            # Compute attention context
            attn_output = self._apply_attention(q, k, v, mask=mask)

            # Reshape back to [batch, seq_len, d_model]
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

            # Output projection & Residual connection 1
            attn_out_projected = torch.matmul(attn_output, self.W_o) + self.b_o
            x = x + attn_out_projected

            # ==========================================
            # 2. Sub-layer 2: Pre-RMSNorm + Feed-Forward Network + Residual Connection
            # ==========================================
            norm_x2 = self.rms_norm_2.forward(x)

            # FFN: Linear -> Activation (GELU) -> Linear
            ffn_hidden = torch.matmul(norm_x2, self.W_fc1) + self.b_fc1
            ffn_activated = torch.gelu(ffn_hidden)
            ffn_output = torch.matmul(ffn_activated, self.W_fc2) + self.b_fc2

            # Residual connection 2
            out = x + ffn_output

            return out

        except Exception as e:
            logger.error(f"Critical error during DecoderTransformerLayer forward pass: {e}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    try:
        logger.info("Starting execution verification of DecoderTransformerLayer...")

        # Hyperparameters
        BATCH_SIZE = 2
        SEQ_LEN_PROMPT = 4
        D_MODEL = 64
        N_HEADS = 4
        D_FF = 256

        # Instantiate layer
        decoder_layer = DecoderTransformerLayer(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF
        )

        # Simulate Prompt Pre-fill Phase
        logger.info("--- Phase 1: Prompt Pre-fill ---")
        prompt_input = torch.randn(BATCH_SIZE, SEQ_LEN_PROMPT, D_MODEL)
        logger.info(f"Input prompt tensor shape: {prompt_input.shape}")

        prefill_output = decoder_layer.forward(prompt_input, use_cache=True)
        logger.info(f"Pre-fill output tensor shape: {prefill_output.shape}")

        # Simulate Auto-regressive Generation Phase (Token-by-Token)
        logger.info("--- Phase 2: Auto-regressive Generation (KV Caching Active) ---")
        NUM_GENERATION_STEPS = 3
        
        for step in range(NUM_GENERATION_STEPS):
            next_token_input = torch.randn(BATCH_SIZE, 1, D_MODEL)
            logger.info(f"Generation Step {step + 1}: Input shape {next_token_input.shape}")

            step_output = decoder_layer.forward(next_token_input, use_cache=True)
            logger.info(f"Generation Step {step + 1}: Output shape {step_output.shape}")

        logger.info("All verification phases completed successfully without errors.")

    except Exception as e:
        logger.error(f"Execution failed in main block: {e}")
        traceback.print_exc()
        sys.exit(1)