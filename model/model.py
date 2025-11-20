import math
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from typing import Optional, Tuple

# ==========================================
# 1. 配置类 (Configuration)
# ==========================================
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_base: float = 1e6,  # 对应 rope_theta
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        # ============ MoE 参数 ============
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        """
        MokioMind 模型的配置类，继承自 Hugging Face PretrainedConfig
        """
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        
        # MoE 参数绑定
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        # YaRN RoPE 缩放配置 (仅在推理开启时生效)
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


# ==========================================
# 2. RMSNorm 层
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Root Mean Square Layer Normalization
        参数:
            dim: 输入特征维度
            eps: 防止除零的极小值 (Epsilon)
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 可学习的缩放参数 gamma，初始化为 1
        self.weights = nn.Parameter(torch.ones(dim))

    def _norm_(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行核心归一化计算：x * rsqrt(mean(x^2))
        """
        # 强制转为 FP32 进行统计量计算，防止溢出
        x_float = x.float()
        
        # 1. 计算 x^2 的均值
        # 2. 加上 eps
        # 3. 开倒数平方根 (1/sqrt)
        inv_rms = torch.rsqrt(torch.mean(x_float.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # 归一化
        return x_float * inv_rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：混合精度处理
        """
        # 如果输入已经是 FP32，直接计算
        if x.dtype == torch.float32:
            normed = self._norm_(x)
        else:
            # 否则：升格计算 norm -> 降格回原类型
            normed = self._norm_(x).type_as(x)
        
        # 最后乘上可学习参数 weights
        return self.weights * normed


# ==========================================
# 3. RoPE (旋转位置编码) 工具函数
# ==========================================

def precompute_freqs_cis(
    dim: int,
    end: int = 32 * 1024, 
    rope_theta: float = 1000000.0,
    rope_scaling: Optional[dict] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 Cos 和 Sin 频率表。
    包含 YaRN (Yet another RoPE for Nanogpt) 的长文本外推逻辑。
    """
    # 1. 计算基础频率 theta_i = base^(-2i/d)
    # shape: (dim/2)
    freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))

    # 2. 如果配置了 rope_scaling (YaRN)，则调整频率
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 4)
        beta_fast = rope_scaling.get("beta_fast", 32)
        beta_slow = rope_scaling.get("beta_slow", 1)
        
        # 仅当需要外推时才执行 (当前长度 > 原始训练长度)
        # 注意：通常 YaRN 会始终应用以保持一致性，这里保留你的判断逻辑
        if end / orig_max > 1.0:
            # 找到高频和低频的分界点 dim index
            # next() 寻找第一个波长超过原始长度的维度
            try:
                corr_dim = next(i for i in range(dim//2) if 2 * math.pi / freqs[i] > orig_max)
            except StopIteration:
                corr_dim = dim // 2

            # 计算平滑插值因子 (Ramp function)
            power = torch.arange(0, dim//2, device=freqs.device).float() / (max(dim//2 - 1, 1))
            beta = beta_slow + (beta_fast - beta_slow) * power
            
            # 根据维度频率不同，应用不同的缩放策略
            scale = torch.where(
                torch.arange(0, dim//2, device=freqs.device) < corr_dim,
                # 高频部分：不怎么变 (接近 1)
                (beta * factor - beta + 1) / (beta * factor),
                # 低频部分：直接按 factor 缩放 (1/factor)
                1.0 / factor
            )
            # 应用缩放
            freqs = freqs * scale

    # 3. 生成时间步 t: [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device).float()
    
    # 4. 外积计算所有位置的角度: time * freq
    # shape: (end, dim/2)
    freqs = torch.outer(t, freqs)

    # 5. 生成 Cos 和 Sin 表
    # 为了配合 rotate_half (切分前后半) 的实现，这里使用 cat 进行拼接
    # 结果形状: (end, dim)
    # 逻辑: [cos(theta_0), ..., cos(theta_d/2), cos(theta_0), ..., cos(theta_d/2)]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在前向传播中实际应用 RoPE。
    
    参数:
        q, k: (Batch, Seq_Len, Head, Dim)
        cos, sin: (Seq_Len, Dim) - 来自 precompute_freqs_cis
        unsqueeze_dim: 用于广播 cos/sin 的维度，通常为 1 (Head 维度) 或 0 (Batch 维度)
    """
    
    def rotate_half(x):
        """
        将 x 切分为两半: x1, x2
        返回: [-x2, x1]
        这是复数旋转 x * i 的实数域实现
        """
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    # 调整 cos/sin 形状以支持广播
    # 假设 q 是 [B, Seq, Head, Dim]，cos 是 [Seq, Dim]
    # unsqueeze(1) 后 cos 变成 [Seq, 1, Dim]，可以广播到 Head 维度
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # 核心旋转公式:
    # x_rot = (x * cos) + (rotate_half(x) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed