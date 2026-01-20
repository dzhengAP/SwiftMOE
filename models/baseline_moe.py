"""
Standard PyTorch MoE implementation without optimizations.
This serves as the baseline for comparison.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple, Optional


class StandardMoE(nn.Module):
    """
    Baseline Mixture-of-Experts implementation.
    - No kernel fusion
    - Standard PyTorch all-to-all
    - No NCCL optimization
    """
    
    def __init__(
        self, 
        num_experts: int = 128, 
        d_model: int = 1024, 
        use_fp16: bool = True,
        world_size: Optional[int] = None
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.use_fp16 = use_fp16
        
        if world_size is None:
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        else:
            self.world_size = world_size
            
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.experts_per_rank = num_experts // self.world_size
        
        dtype = torch.float16 if use_fp16 else torch.float32
        
        # Gate network (shared across all ranks)
        self.gate = nn.Linear(d_model, num_experts, bias=False, dtype=dtype)
        
        # Local experts
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False, dtype=dtype)
            for _ in range(self.experts_per_rank)
        ])
        
        # Initialize weights
        nn.init.normal_(self.gate.weight, std=0.02)
        for expert in self.experts:
            nn.init.normal_(expert.weight, std=0.02)
    
    def _route_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard routing without fusion"""
        # Gate computation
        gate_logits = self.gate(x)  # [batch, num_experts]
        
        # Top-2 selection
        top_k_logits, top_k_indices = torch.topk(gate_logits, k=2, dim=-1)
        
        # Softmax over top-k
        top_k_weights = torch.softmax(top_k_logits, dim=-1)
        
        return top_k_indices, top_k_weights
    
    def _dispatch_tokens(
        self, 
        x: torch.Tensor, 
        indices: torch.Tensor, 
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, list, torch.Tensor]:
        """Standard dispatch using all-to-all"""
        batch_size = x.shape[0]
        
        # Flatten: [batch, 2] -> [batch*2]
        flat_indices = indices.view(-1)
        flat_weights = weights.view(-1, 1)
        
        # Compute target ranks
        target_ranks = flat_indices // self.experts_per_rank
        local_indices = flat_indices % self.experts_per_rank
        
        # Sort by target rank
        sorted_indices = torch.argsort(target_ranks)
        reverse_indices = torch.argsort(sorted_indices)
        
        sorted_x = x.repeat_interleave(2, dim=0)[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_local_idx = local_indices[sorted_indices]
        
        # Count tokens per rank - KEEP ON GPU
        send_counts = torch.bincount(
            target_ranks, 
            minlength=self.world_size
        )
        
        # All-to-all for counts - KEEP ON GPU
        recv_counts = torch.empty_like(send_counts)
        if dist.is_initialized():
            dist.all_to_all_single(recv_counts, send_counts)
        else:
            recv_counts = send_counts
        
        total_recv = recv_counts.sum().item()
        
        # All-to-all for data
        recv_x = torch.empty(
            (total_recv, self.d_model), 
            device=x.device, 
            dtype=x.dtype
        )
        recv_weights = torch.empty(
            (total_recv, 1), 
            device=x.device, 
            dtype=x.dtype
        )
        recv_local_idx = torch.empty(
            (total_recv,), 
            device=x.device, 
            dtype=torch.long
        )
        
        if dist.is_initialized():
            # Convert to list AFTER communication for split sizes
            send_list = send_counts.cpu().tolist()
            recv_list = recv_counts.cpu().tolist()
            
            dist.all_to_all_single(recv_x, sorted_x, recv_list, send_list)
            dist.all_to_all_single(recv_weights, sorted_weights, recv_list, send_list)
            dist.all_to_all_single(recv_local_idx, sorted_local_idx, recv_list, send_list)
        else:
            recv_x = sorted_x
            recv_weights = sorted_weights
            recv_local_idx = sorted_local_idx
            send_list = send_counts.cpu().tolist()
            recv_list = recv_counts.cpu().tolist()
        
        return recv_x, recv_local_idx, recv_weights, send_list, recv_list, reverse_indices
    
    def _expert_forward(
        self, 
        recv_x: torch.Tensor, 
        recv_idx: torch.Tensor, 
        recv_weights: torch.Tensor
    ) -> torch.Tensor:
        """Execute local experts"""
        output = torch.zeros_like(recv_x)
        
        for i in range(self.experts_per_rank):
            mask = (recv_idx == i)
            if mask.any():
                expert_input = recv_x[mask]
                expert_weights = recv_weights[mask]
                
                # Expert computation
                expert_output = self.experts[i](expert_input)
                
                # Weight and accumulate
                output[mask] = expert_output * expert_weights
        
        return output
    
    def _combine_tokens(
        self, 
        expert_output: torch.Tensor, 
        send_list: list, 
        recv_list: list, 
        reverse_indices: torch.Tensor,
        original_batch_size: int
    ) -> torch.Tensor:
        """Combine expert outputs"""
        # All-to-all to return results
        combined = torch.empty(
            (sum(send_list), self.d_model), 
            device=expert_output.device, 
            dtype=expert_output.dtype
        )
        
        if dist.is_initialized():
            dist.all_to_all_single(combined, expert_output, send_list, recv_list)
        else:
            combined = expert_output
        
        # Restore original order and sum top-2
        restored = combined[reverse_indices].view(original_batch_size, 2, self.d_model)
        return restored.sum(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        
        original_batch_size = x.shape[0]
        
        # Route
        indices, weights = self._route_tokens(x)
        
        # Dispatch
        recv_x, recv_idx, recv_weights, send_list, recv_list, rev_idx = \
            self._dispatch_tokens(x, indices, weights)
        
        # Expert computation
        expert_output = self._expert_forward(recv_x, recv_idx, recv_weights)
        
        # Combine
        output = self._combine_tokens(
            expert_output, send_list, recv_list, rev_idx, original_batch_size
        )
        
        return output
