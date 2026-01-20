"""
DeepSpeed-MoE wrapper for fair comparison.
Requires: pip install deepspeed
"""

import torch
import torch.nn as nn
try:
    import deepspeed
    from deepspeed.moe.layer import MoE as DeepSpeedMoE
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("⚠️ DeepSpeed not available. Install with: pip install deepspeed")


class DeepSpeedMoEWrapper(nn.Module):
    """
    Wrapper around DeepSpeed MoE for benchmarking.
    Matches configuration with UltimateMoE for fair comparison.
    """
    
    def __init__(
        self, 
        num_experts: int = 128,
        d_model: int = 1024,
        expert_parallel_size: int = None,
        use_fp16: bool = True
    ):
        super().__init__()
        
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is required for this comparison")
        
        self.num_experts = num_experts
        self.d_model = d_model
        self.use_fp16 = use_fp16
        
        if expert_parallel_size is None:
            if torch.distributed.is_initialized():
                expert_parallel_size = torch.distributed.get_world_size()
            else:
                expert_parallel_size = 1
        
        self.ep_size = expert_parallel_size
        
        # Create expert module (simple linear layer)
        dtype = torch.float16 if use_fp16 else torch.float32
        expert = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        
        # Initialize DeepSpeed MoE layer
        self.moe_layer = DeepSpeedMoE(
            hidden_size=d_model,
            expert=expert,
            num_experts=num_experts,
            k=2,  # Top-2 routing
            ep_size=expert_parallel_size,
            use_residual=False,
            min_capacity=0,  # No capacity factor
            noisy_gate_policy=None,  # No noise
            drop_tokens=False,  # Don't drop tokens
            use_rts=True,  # Use optimized routing
            use_tutel=False  # Standard DeepSpeed implementation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepSpeed MoE.
        Returns only the output (discards auxiliary losses).
        """
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        
        # DeepSpeed MoE returns (output, moe_loss, router_probs)
        output, _, _ = self.moe_layer(x)
        
        return output


class MegatronMoEWrapper(nn.Module):
    """
    Megatron-style MoE implementation.
    Based on Megatron-LM architecture but simplified for comparison.
    """
    
    def __init__(
        self,
        num_experts: int = 128,
        d_model: int = 1024,
        use_fp16: bool = True,
        use_expert_parallel: bool = True
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.d_model = d_model
        self.use_fp16 = use_fp16
        
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        
        self.experts_per_rank = num_experts // self.world_size if use_expert_parallel else num_experts
        
        dtype = torch.float16 if use_fp16 else torch.float32
        
        # Router (gating network)
        self.gate = nn.Linear(d_model, num_experts, bias=False, dtype=dtype)
        
        # Local experts
        self.experts = nn.ModuleList([
            self._create_expert(d_model, dtype)
            for _ in range(self.experts_per_rank)
        ])
        
        # Load balancing
        self.expert_capacity_factor = 1.0
        
    def _create_expert(self, d_model, dtype):
        """Create Megatron-style expert (MLP)"""
        return nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False, dtype=dtype),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        
        batch_size, seq_len, hidden_dim = x.shape[0], 1, x.shape[1]
        x_flat = x.view(-1, hidden_dim)
        
        # Routing
        router_logits = self.gate(x_flat)
        routing_weights, selected_experts = torch.topk(router_logits, k=2, dim=-1)
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        # Dispatch and compute (simplified - no actual expert parallelism in this wrapper)
        output = torch.zeros_like(x_flat)
        
        for i in range(self.experts_per_rank):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == i).any(dim=-1)
            
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[i](expert_input)
                
                # Apply routing weights
                weights = routing_weights[expert_mask][selected_experts[expert_mask] == i]
                output[expert_mask] += expert_output * weights.unsqueeze(-1)
        
        return output.view(batch_size, seq_len, hidden_dim).squeeze(1)
