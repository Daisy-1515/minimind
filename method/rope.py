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