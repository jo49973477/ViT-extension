import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

from scipy import ndimage


# ------------- MULTI-HEAD ATTENTION -----------------
class Attention(nn.Module):
    """ Vanilla Multi-Head Attention Network
        hidden state = (B, L, H) : B=batch size, L=tokens, H=hidden states length
        attention_output =  (B, L, H) : B=batch size, L=tokens, H=hidden states length
    """
    
    
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


# ------------- ROTARY EMBEDDING FUNCTION -----------------
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    config:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    
    return y.to(dtype)


# ------------- RMS Normalization -----------------
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    config:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        config:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


# ------------- MULTI-HEAD LATENT ATTENTION -----------------
class MultiHeadLatentAttentionViT(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.
    (ViT-compatible version by Mamang 💖)
    
    - No KV Caching (ViT processes all tokens at once)
    - No RoPE (Assumes ViT's Absolute Position Embeddings (APE) are already added to 'x')
    - No Quantization (Removed for training/research focus)
    """
    
    def __init__(self, config, atten_naive: bool = False):
        super().__init__()
        self.atten_naive = atten_naive 
        
        self.dim = config.hidden_size
        self.n_heads = config.transformer["num_heads"]
        self.head_dim = self.dim // self.n_heads
        self.v_head_dim = self.head_dim 
        
        self.kv_lora_rank = config.transformer.get("kv_lora_rank", 128) 
        
        self.wq = Linear(self.dim, self.n_heads * self.head_dim)
        
        # --- MLA 핵심 로직 (K, V 압축) ---
        self.wkv_a = Linear(self.dim, self.kv_lora_rank) 
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        
        self.wkv_b = Linear(self.kv_lora_rank, self.n_heads * (self.head_dim + self.v_head_dim)) 
        
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.head_dim ** -0.5
        

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        """
        Forward pass (ViT-compatible)
        - No 'start_pos', 'freqs_cis'
        - 'x' (B, S, D)는 모든 토큰(197개)을 한 번에 받음
        """
        bsz, seqlen, _ = x.size() 

        # 1. Q 계산 (RoPE 로직 없음)
        q = self.wq(x) # (B, S, n_heads * head_dim)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3) # (B, n_heads, S, head_dim)

        # 2. KV 잠재 벡터 계산 (RoPE 로직 없음)
        kv_latent = self.wkv_a(x) # (B, S, kv_lora_rank)
        kv_latent_norm = self.kv_norm(kv_latent) # (B, S, kv_lora_rank)

        if self.atten_naive:
            # --- 🐢 Naive 모드 (압축 풀고 계산) ---
            
            # 3a. K, V를 '즉시' 압축 해제
            kv = self.wkv_b(kv_latent_norm) # (B, S, n_heads * (k_dim + v_dim))
            kv = kv.view(bsz, seqlen, self.n_heads, self.head_dim + self.v_head_dim)
            k, v = torch.split(kv, [self.head_dim, self.v_head_dim], dim=-1)
            
            k = k.permute(0, 2, 1, 3) # (B, n_heads, S, head_dim)
            v = v.permute(0, 2, 1, 3) # (B, n_heads, S, v_head_dim)

            # 4a. 어텐션 계산 (캐시 사용 X, 'k'와 'v' 직접 사용)
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
            if mask is not None:
                scores += mask
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            output = torch.matmul(scores, v) # (B, n_heads, S, v_head_dim)

        else:
            
            # 3b. 압축 해제기 '가중치'만 가져오기 (양자화 로직 제외)
            wkv_b_weight = self.wkv_b.weight
            
            # (R, H_all) -> (H, D_kv, R)
            wkv_b_weight = wkv_b_weight.view(self.kv_lora_rank, self.n_heads, self.head_dim + self.v_head_dim)
            wkv_b_weight = wkv_b_weight.permute(1, 2, 0) # (n_heads, D_kv, R)

            # K용 가중치, V용 가중치 분리
            k_weight, v_weight = torch.split(wkv_b_weight, [self.head_dim, self.v_head_dim], dim=1)
            
            # 4b. Q를 Latent 공간으로 프로젝션 (einsum 사용)
            # q: (B, H, S, D), k_weight: (H, D, R) -> q_latent: (B, H, S, R)
            q_latent = torch.einsum("bhsd,hdr->bhsr", q, k_weight)
            
            # 5b. 어텐션 스코어 계산 (Latent 공간에서!)
            # q_latent: (B, H, S, R), kv_latent_norm: (B, T, R) -> (B, H, S, T)
            scores = torch.einsum("bhsr,btr->bhst", q_latent, kv_latent_norm) * self.softmax_scale
            if mask is not None:
                scores += mask
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            
            # 6b. 출력 계산 (Latent 공간에서!)
            # scores: (B, H, S, S), kv_latent_norm: (B, S, R) -> (B, H, S, R)
            output_latent = torch.einsum("bhst,btr->bhsr", scores, kv_latent_norm)
            
            # 7b. 최종 출력 '즉시' 복원 (einsum 사용)
            # output_latent: (B, H, S, R), v_weight: (H, D, R) -> (B, H, S, D)
            output = torch.einsum("bhsr,hdr->bhsd", output_latent, v_weight)

        # --- 최종 출력 (공통) ---
        # (B, n_heads, S, v_head_dim) -> (B, S, n_heads * v_head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        output = self.wo(output) # (B, S, dim)
        return output