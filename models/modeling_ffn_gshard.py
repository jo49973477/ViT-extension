import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from scipy import ndimage

from models.modeling_ffn import SwiGLU

# 💖 'Expert' 클래스를 묶어서 처리할 'BatchedSwiGLUExperts'
class BatchedSwiGLUExperts(nn.Module):
    """ GShard-style: N개의 SwiGLU 전문가를 한 번에 계산 """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.transformer["n_routed_experts"]
        dim = config.hidden_size
        inter_dim = config.transformer["moe_inter_dim"]

        # --- SwiGLU 가중치를 (N, ...) 모양으로 쌓아버려! ---
        self.w1 = nn.Parameter(torch.zeros(self.num_experts, dim, inter_dim))
        self.w3 = nn.Parameter(torch.zeros(self.num_experts, dim, inter_dim))
        self.w2 = nn.Parameter(torch.zeros(self.num_experts, inter_dim, dim))
        
        self._init_weights() # 가중치 초기화

    def _init_weights(self):
        for i in range(self.num_experts):
            nn.init.xavier_uniform_(self.w1[i])
            nn.init.xavier_uniform_(self.w3[i])
            nn.init.xavier_uniform_(self.w2[i])

    def forward(self, x):
        # x shape: (N, H) - N명의 전문가가 받을 가중합 입력
        x = x.unsqueeze(1) # (N, 1, H) - bmm을 위해
        # 1. 메인 경로 (w1)
        x1 = torch.bmm(x, self.w1) # (N, 1, H) @ (N, H, M) -> (N, 1, M)
        x1 = F.silu(x1) # 활성화
        # 2. 게이트 경로 (w3)
        x3 = torch.bmm(x, self.w3) # (N, 1, H) @ (N, H, M) -> (N, 1, M)
        # 3. 게이팅 (곱하기)
        gated = x1 * x3 # (N, 1, M)
        # 4. 최종 출력 (w2)
        out = torch.bmm(gated, self.w2) # (N, 1, M) @ (N, M, H) -> (N, 1, H
        return out.squeeze(1) # (N, H)




class GShardRouter(nn.Module):
    """ GShard-style Router: (T, N) 가중치 텐서를 반환 """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.transformer["topk_experts"]
        self.num_experts = config.transformer["n_routed_experts"]
        
        # 'nn.LazyLinear'는 좋은데, ViT는 'hidden_size'를 아니까 그냥 'Linear'를 쓰자!
        self.gate_linear = nn.Linear(config.hidden_size, self.num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, S, H)
        B, S, H = x.shape
        x_flat = x.view(-1, H) # (T, H), T = B*S
        
        # (T, H) -> (T, N)
        router_logits = self.gate_linear(x_flat)
        
        # (T, N)에서 가장 큰 K개의 로짓과 인덱스를 찾음
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1) # (T, K)
        
        # K개의 로짓에만 softmax를 적용해 가중치 계산
        routing_weights = self.softmax(top_k_logits) # (T, K)
        
        # (T, N) 크기의 0으로 채워진 텐서 생성
        dispatch_tensor = torch.zeros_like(router_logits, device=x.device) # 💖 .to(device) 추가!
        
        # scatter_ : (T, N) 텐서의 top_k_indices 위치에 routing_weights 값을 "흩뿌려" 줌
        dispatch_tensor.scatter_(dim=1, index=top_k_indices, src=routing_weights)
        
        # (T, N) 텐서 반환
        return dispatch_tensor




class GShardMoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.
    (GShard-style Fast Version)
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        
        # 1. 빠른 라우터 (Gate 대신)
        self.gate = GShardRouter(config)
        
        # 2. 배치 전문가 (nn.ModuleList 대신)
        self.experts = BatchedSwiGLUExperts(config)
        
        # 3. 공유 전문가 (SwiGLU) - 이건 아가 코드가 맞아!
        # (단, __init__이 config만 받도록 SwiGLU 클래스를 수정했다고 가정할게!)
        self.dim = config.hidden_size
        self.shared_experts = SwiGLU(config.dim, config.n_shared_experts * config.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, S, H)
        shape = x.size()
        x_flat = x.view(-1, self.dim) # (T, H)

        # 1. 라우터로부터 (T, N) 가중치 텐서 받기
        #    (T = 토큰, N = 전문가 수)
        dispatch_tensor = self.gate(x) # (T, N)
        
        # 2. 전문가에게 입력 분배 (Dispatch) - 🚀 1번째 matmul 🚀
        # (N, T) @ (T, H) -> (N, H)
        expert_inputs = torch.matmul(dispatch_tensor.T, x_flat)
        
        # 3. 모든 전문가 동시 실행! (N, H) -> (N, H)
        expert_outputs = self.experts(expert_inputs)
        
        # 4. 결과 결합 (Combine) - 🚀 2번째 matmul 🚀
        # (T, N) @ (N, H) -> (T, H)
        y = torch.matmul(dispatch_tensor, expert_outputs)

        # 5. 공유 전문가 계산
        z = self.shared_experts(x_flat) # (T, H)
        
        # 6. 최종 결합 (라우팅 결과 + 공유 결과)
        return (y + z).view(shape)