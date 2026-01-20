"""
Optimized UltimateMoE implementation with instrumentation.
This is your optimized version with detailed profiling hooks.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import triton
import triton.language as tl
from typing import Dict, Optional, Tuple


def setup_nccl_optimizations():
    """Configure NCCL for optimal performance"""
    nccl_config = {
        'NCCL_IB_DISABLE': '0',
        'NCCL_P2P_LEVEL': 'NVL',
        'NCCL_NET_GDR_LEVEL': '5',
        'NCCL_SOCKET_IFNAME': 'eth0',
        'NCCL_MIN_NCHANNELS': '12',
        'NCCL_BUFFSIZE': '4194304',
        'NCCL_NTHREADS': '512',
    }
    
    for key, value in nccl_config.items():
        if key not in os.environ:
            os.environ[key] = value
    
    return nccl_config


@triton.jit
def fused_router_dispatch_kernel(
    x_ptr, gate_ptr, weight_ptr, idx_ptr,
    target_rank_ptr, local_idx_ptr,
    stride_xm, stride_xk, stride_gn, stride_gk,
    M, K, N, exp_per_rank,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    USE_FP16: tl.constexpr,
):
    """Fused Router + Dispatch kernel"""
    pid = tl.program_id(0)
    rm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)
    
    # FP32 accumulator for numerical stability
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Tiled GEMM
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        x_mask = (rm[:, None] < M) & (rk[None, :] < K)
        g_mask = (rk[:, None] < K) & (rn[None, :] < N)
        
        x_ptrs = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
        g_ptrs = gate_ptr + (rk[:, None] * stride_gk + rn[None, :] * stride_gn)
        
        cur_x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        cur_g = tl.load(g_ptrs, mask=g_mask, other=0.0).to(tl.float32)
        
        acc += tl.dot(cur_x, cur_g)
    
    # Top-2 selection
    m1_val, m1_idx = tl.max(acc, axis=1, return_indices=True)
    mask_top1 = (tl.arange(0, BLOCK_SIZE_N)[None, :] == m1_idx[:, None])
    acc_masked = tl.where(mask_top1, -1e9, acc)
    m2_val, m2_idx = tl.max(acc_masked, axis=1, return_indices=True)
    
    # Softmax normalization
    e1, e2 = tl.exp(m1_val), tl.exp(m2_val)
    sum_e = e1 + e2
    w1, w2 = e1 / sum_e, e2 / sum_e
    
    if USE_FP16:
        w1 = w1.to(tl.float16)
        w2 = w2.to(tl.float16)
    
    # Write routing results
    out_base = rm * 2
    mask_valid = rm < M
    
    tl.store(idx_ptr + out_base + 0, m1_idx, mask=mask_valid)
    tl.store(idx_ptr + out_base + 1, m2_idx, mask=mask_valid)
    tl.store(weight_ptr + out_base + 0, w1, mask=mask_valid)
    tl.store(weight_ptr + out_base + 1, w2, mask=mask_valid)
    
    # Fused dispatch metadata
    rank1 = m1_idx // exp_per_rank
    rank2 = m2_idx // exp_per_rank
    local1 = m1_idx % exp_per_rank
    local2 = m2_idx % exp_per_rank
    
    tl.store(target_rank_ptr + out_base + 0, rank1, mask=mask_valid)
    tl.store(target_rank_ptr + out_base + 1, rank2, mask=mask_valid)
    tl.store(local_idx_ptr + out_base + 0, local1, mask=mask_valid)
    tl.store(local_idx_ptr + out_base + 1, local2, mask=mask_valid)


class MoEAsyncCommHandler:
    """Optimized all-to-all communication handler"""
    
    def __init__(self, num_experts, world_size, d_model, device):
        self.num_experts = num_experts
        self.world_size = world_size
        self.d_model = d_model
        self.device = device
        self.exp_per_rank = num_experts // world_size

    def dispatch(self, x, indices, weights, target_ranks, local_indices):
        """Optimized dispatch with pre-computed ranks"""
        M, K = x.shape
        
        send_counts = torch.bincount(target_ranks, minlength=self.world_size)
        
        sort_idx = torch.argsort(target_ranks)
        sort_idx_rev = torch.argsort(sort_idx)
        
        sorted_x = x.repeat_interleave(2, dim=0)[sort_idx]
        sorted_weights = weights.view(-1, 1)[sort_idx]
        sorted_local_indices = local_indices.view(-1, 1)[sort_idx]
        
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        
        total_recv = recv_counts.sum().item()
        
        recv_x = torch.empty((total_recv, K), device=self.device, dtype=x.dtype)
        recv_weights = torch.empty((total_recv, 1), device=self.device, dtype=x.dtype)
        recv_local_indices = torch.empty((total_recv, 1), device=self.device, dtype=torch.int32)
        
        s_list = send_counts.tolist()
        r_list = recv_counts.tolist()
        
        dist.all_to_all_single(recv_x, sorted_x, r_list, s_list)
        dist.all_to_all_single(recv_weights, sorted_weights, r_list, s_list)
        dist.all_to_all_single(recv_local_indices, sorted_local_indices, r_list, s_list)
        
        return recv_x, recv_local_indices.view(-1), recv_weights, s_list, r_list, sort_idx_rev

    def combine(self, expert_out, s_list, r_list, sort_idx_rev, M):
        """Combine expert outputs"""
        combined_x = torch.empty(
            (sum(s_list), self.d_model), 
            device=self.device, 
            dtype=expert_out.dtype
        )
        dist.all_to_all_single(combined_x, expert_out, s_list, r_list)
        
        restored = combined_x[sort_idx_rev].view(M, 2, self.d_model)
        return restored.sum(dim=1)


class UltimateMoE(nn.Module):
    """
    Optimized Mixture-of-Experts with:
    - Triton kernel fusion
    - NCCL optimization
    - Mixed precision (FP16/FP32)
    - Detailed profiling instrumentation
    """
    
    def __init__(
        self, 
        num_experts: int = 128, 
        d_model: int = 1024, 
        use_fp16: bool = True,
        use_triton: bool = True,
        use_nccl_opt: bool = True,
        enable_profiling: bool = False
    ):
        super().__init__()
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.ws = dist.get_world_size() if dist.is_initialized() else 1
        self.num_experts = num_experts
        self.d_model = d_model
        self.use_fp16 = use_fp16
        self.use_triton = use_triton
        self.use_nccl_opt = use_nccl_opt
        self.enable_profiling = enable_profiling
        self.exp_per_rank = num_experts // self.ws
        
        dtype = torch.float16 if use_fp16 else torch.float32
        
        # Expert weights
        self.expert_weights = nn.Parameter(
            torch.randn(self.exp_per_rank, d_model, d_model, dtype=dtype)
        )
        
        # Gate weight
        self.gate_weight = nn.Parameter(
            torch.randn(num_experts, d_model, dtype=dtype)
        )
        
        # Initialize
        nn.init.normal_(self.expert_weights, std=0.02)
        nn.init.normal_(self.gate_weight, std=0.02)
        
        # Communication handler
        device = f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        self.comm = MoEAsyncCommHandler(num_experts, self.ws, d_model, device)
        
        # Compute stream for overlap
        self.compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Profiling storage
        self.profile_data = {
            'router': [],
            'dispatch': [],
            'compute': [],
            'combine': []
        }
        
        # Apply NCCL optimizations
        if use_nccl_opt:
            setup_nccl_optimizations()

    def _expert_compute(self, recv_x, recv_idx, recv_w):
        """Expert forward computation"""
        expert_order = torch.argsort(recv_idx)
        reordered_x = recv_x[expert_order]
        reordered_w = recv_w[expert_order]
        
        tokens_per_exp = torch.bincount(
            recv_idx, 
            minlength=self.exp_per_rank
        ).cpu().tolist()
        
        expert_out = torch.empty_like(reordered_x)
        curr_ptr = 0
        
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            for i in range(self.exp_per_rank):
                num = tokens_per_exp[i]
                if num == 0:
                    continue
                
                expert_input = reordered_x[curr_ptr : curr_ptr + num]
                expert_weight = reordered_w[curr_ptr : curr_ptr + num]
                
                weighted_input = expert_input * expert_weight
                expert_out[curr_ptr : curr_ptr + num] = torch.mm(
                    weighted_input, 
                    self.expert_weights[i]
                )
                curr_ptr += num
        
        return expert_out[torch.argsort(expert_order)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional profiling"""
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        
        M, K = x.shape
        
        # Initialize buffers
        indices = torch.zeros((M, 2), device=x.device, dtype=torch.int32)
        weights = torch.zeros((M, 2), device=x.device, dtype=x.dtype)
        target_ranks = torch.zeros((M, 2), device=x.device, dtype=torch.int32)
        local_indices = torch.zeros((M, 2), device=x.device, dtype=torch.int32)
        
        # Router stage
        if self.enable_profiling:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        if self.use_triton:
            grid = (triton.cdiv(M, 64),)
            fused_router_dispatch_kernel[grid](
                x, self.gate_weight, weights, indices,
                target_ranks, local_indices,
                x.stride(0), x.stride(1),
                self.gate_weight.stride(0), self.gate_weight.stride(1),
                M, K, self.num_experts, self.exp_per_rank,
                BLOCK_SIZE_M=64, 
                BLOCK_SIZE_N=128, 
                BLOCK_SIZE_K=32,
                USE_FP16=self.use_fp16
            )
        else:
            # Fallback to standard routing
            gate_logits = torch.matmul(x, self.gate_weight.t())
            top_k_logits, top_k_indices = torch.topk(gate_logits, k=2, dim=-1)
            top_k_weights = torch.softmax(top_k_logits, dim=-1)
            
            indices = top_k_indices.int()
            weights = top_k_weights
            target_ranks = indices // self.exp_per_rank
            local_indices = indices % self.exp_per_rank
        
        if self.enable_profiling:
            end_event.record()
            torch.cuda.synchronize()
            self.profile_data['router'].append(start_event.elapsed_time(end_event))
        
        # Dispatch stage
        if self.enable_profiling:
            start_event.record()
        
        recv_x, recv_idx, recv_w, s_list, r_list, rev_idx = self.comm.dispatch(
            x, indices, weights, target_ranks.view(-1), local_indices.view(-1)
        )
        
        if self.enable_profiling:
            end_event.record()
            torch.cuda.synchronize()
            self.profile_data['dispatch'].append(start_event.elapsed_time(end_event))
        
        # Compute stage
        if self.enable_profiling:
            start_event.record()
        
        if self.compute_stream is not None:
            with torch.cuda.stream(self.compute_stream):
                expert_out = self._expert_compute(recv_x, recv_idx, recv_w)
            torch.cuda.current_stream().wait_stream(self.compute_stream)
        else:
            expert_out = self._expert_compute(recv_x, recv_idx, recv_w)
        
        if self.enable_profiling:
            end_event.record()
            torch.cuda.synchronize()
            self.profile_data['compute'].append(start_event.elapsed_time(end_event))
        
        # Combine stage
        if self.enable_profiling:
            start_event.record()
        
        output = self.comm.combine(expert_out, s_list, r_list, rev_idx, M)
        
        if self.enable_profiling:
            end_event.record()
            torch.cuda.synchronize()
            self.profile_data['combine'].append(start_event.elapsed_time(end_event))
        
        return output
    
    def get_profile_summary(self) -> Dict[str, float]:
        """Get averaged profiling statistics"""
        import numpy as np
        return {
            'Router': np.mean(self.profile_data['router']) if self.profile_data['router'] else 0,
            'Dispatch_Comm': np.mean(self.profile_data['dispatch']) if self.profile_data['dispatch'] else 0,
            'Expert_Compute': np.mean(self.profile_data['compute']) if self.profile_data['compute'] else 0,
            'Combine_Comm': np.mean(self.profile_data['combine']) if self.profile_data['combine'] else 0
        }
    
    def reset_profiling(self):
        """Clear profiling data"""
        for key in self.profile_data:
            self.profile_data[key].clear()
