"""
MokioMind 语言模型实现

这是一个基于 Transformer 架构的因果语言模型 (Causal Language Model)，
采用了多项现代优化技术：
- GQA (Grouped Query Attention): 分组查询注意力，减少 KV Cache 显存占用
- RoPE (Rotary Position Embedding): 旋转位置编码，支持长文本外推
- RMSNorm: Root Mean Square 层归一化，计算更高效
- SwiGLU: 门控线性单元激活函数，提升模型表达能力
- Flash Attention: 融合注意力算子，加速计算并节省显存

作者: MokioMind Team
版本: 1.0
"""

# ==========================================
# 导入依赖库
# ==========================================
from ast import arg
from filecmp import BUFSIZE
from logging import config
import math  # 数学运算库
import re
from turtle import forward

# PyTorch 核心库
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hugging Face Transformers 库
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN  # 激活函数映射字典 {"silu": SiLU(), "gelu": GELU(), ...}

# 类型注解
from typing import Optional, Tuple, List, Union

# 自定义模块
from method import rope  # RoPE 位置编码实现

# ==========================================
# 1. 配置类 (Configuration)
# ==========================================
class MokioMindConfig(PretrainedConfig):
    """
    MokioMind 模型配置类

    继承自 Hugging Face PretrainedConfig，用于存储模型的所有超参数。
    支持自动序列化/反序列化，可与 Hugging Face Hub 无缝集成。

    主要参数分类：
    1. 基础训练参数：dropout
    2. 特殊 Token：bos_token_id, eos_token_id
    3. 模型架构：hidden_size, num_layers, num_heads 等
    4. 位置编码：rope_base, max_position_embeddings
    5. 性能优化：flash_attention
    6. MoE 专家混合：use_moe, num_experts 等

    使用示例：
        >>> config = MokioMindConfig(hidden_size=768, num_attention_heads=12)
        >>> model = MokioMindForCausalLM(config)
    """
    model_type = "mokiomind"  # 模型类型标识符，用于 Hugging Face Hub 注册

    def __init__(
        self,
        # ============ 基础训练参数 ============
        dropout: float = 0.0,  # Dropout 概率，用于防止过拟合（0.0=不使用，0.1-0.3=常见训练值）

        # ============ 特殊 Token ID ============
        bos_token_id: int = 1,  # 序列开始标记 (Begin Of Sentence)，用于标识输入起始
        eos_token_id: int = 2,  # 序列结束标记 (End Of Sentence)，生成此 token 时停止

        # ============ 模型架构参数 ============
        hidden_act: str = "silu",  # 激活函数类型："silu"(Swish)/"gelu"(BERT常用)/"relu"(传统)

        hidden_size: int = 512,  # 模型隐藏层维度，决定表达能力（512=小模型, 768=BERT-base, 4096=Llama-7B）
        intermediate_size: Optional[int] = None,  # FFN 中间层维度，None 时自动计算为 hidden_size*8/3 向上取64倍数
        max_position_embeddings: int = 32768,  # 最大序列长度（32K tokens ≈ 24K 英文单词）
        num_attention_heads: int = 8,  # Query 注意力头数，必须能被 num_key_value_heads 整除
        num_hidden_layers: int = 8,  # Transformer 层数，决定模型深度（6-12=小模型, 24=BERT-large, 32=Llama-7B）
        num_key_value_heads: int = 2,  # KV 注意力头数，用于 GQA（<num_attention_heads 时启用分组查询）

        # ============ 词表参数 ============
        vocab_size: int = 6400,  # 词表大小，决定可识别的 token 数量（32K=Llama, 50K=GPT-2）

        # ============ 归一化参数 ============
        rms_norm_eps: float = 1e-05,  # RMSNorm 的 epsilon 值，防止除零（通常 1e-5 或 1e-6）

        # ============ 位置编码参数 ============
        rope_base: float = 1e6,  # RoPE 基频参数 theta（1e6=超长上下文, 1e4=原始论文值）
        inference_rope_scaling: bool = False,  # 是否启用 YaRN 缩放，支持超过训练长度的推理

        # ============ 性能优化参数 ============
        flash_attention: bool = True,  # 是否启用 Flash Attention（需 PyTorch>=2.0，2-4x 加速）

        # ============ MoE (Mixture of Experts) 参数 ============
        use_moe: bool = False,  # 是否启用混合专家模型（提升容量但保持计算量）
        num_experts_per_tok: int = 2,  # 每个 token 激活的专家数（通常为2，平衡性能和计算）
        n_routed_experts: int = 4,  # 可路由的专家总数（常见值: 4, 8, 16, 64）
        n_shared_experts: int = 1,  # 共享专家数（所有 token 都经过，学习通用特征）
        scoring_func: str = 'softmax',  # 专家选择评分函数（"softmax"=概率分布, "sigmoid"=独立评分）
        aux_loss_alpha: float = 0.1,  # 辅助损失权重，用于平衡专家负载
        seq_aux: bool = True,  # 是否使用序列级辅助损失（True=整个序列, False=每个token独立）
        norm_topk_prob: bool = True,  # 是否归一化 top-k 专家概率（True=重新归一化权重）
        **kwargs,
    ):
        """
        初始化 MokioMind 配置

        参数说明：
            dropout: Dropout 概率，0.0 表示不使用
            hidden_size: 模型维度，必须能被 num_attention_heads 整除
            num_attention_heads: Query 头数，必须能被 num_key_value_heads 整除
            num_key_value_heads: KV 头数，实现 GQA 以减少显存占用

        注意事项：
            - hidden_size 必须能被 num_attention_heads 整除
            - num_attention_heads 必须能被 num_key_value_heads 整除
            - intermediate_size 为 None 时会自动计算
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
    rope_base: float = 1000000.0,
    rope_scaling: Optional[dict] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 Cos 和 Sin 频率表。
    包含 YaRN (Yet another RoPE for Nanogpt) 的长文本外推逻辑。
    """
    # 1. 计算基础频率 theta_i = base^(-2i/d)
    # shape: (dim/2)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))

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
    应用旋转位置编码 (Rotary Position Embedding, RoPE) 到查询和键张量。

    RoPE 通过旋转变换将位置信息编码到注意力机制中,相比传统的绝对位置编码:
    - 能够自然地表达相对位置关系
    - 对序列长度具有更好的外推能力
    - 计算效率高,仅需逐元素乘法和加法
    
    参数:
        q: 查询张量,形状 (Batch, Seq_Len, Head, Dim)
        k: 键张量,形状 (Batch, Seq_Len, Head, Dim)
        cos: 预计算的余弦值,形状 (Seq_Len, Dim),来自 precompute_freqs_cis
        sin: 预计算的正弦值,形状 (Seq_Len, Dim),来自 precompute_freqs_cis
        unsqueeze_dim: 用于广播 cos/sin 的维度索引
                       - 1: 在 Head 维度插入,适用于 (Batch, Seq, Head, Dim) 布局
                       - 0: 在 Batch 维度插入,适用于其他布局

    返回:
        (q_embed, k_embed): 应用 RoPE 后的查询和键张量,形状与输入相同
    """
    
    def rotate_half(x):
        """
        实现复数旋转的辅助函数,将向量旋转 90 度。

        将输入张量的最后一维切分为两半 [x1, x2],返回 [-x2, x1]。
        这对应于复数乘法 x * i 在实数域的表示:
        - 如果 x = x1 + i*x2,则 x * i = -x2 + i*x1

        参数:
            x: 输入张量,最后一维的大小必须是偶数

        返回:
            旋转后的张量,形状与输入相同
        """
        # 将最后一维切分为前半部分和后半部分
        x1 = x[..., :x.shape[-1]//2] # 前半部分: 维度 [0, dim/2)
        x2 = x[..., x.shape[-1]//2:] # 后半部分: 维度 [dim/2, dim)
        # 拼接 [-x2, x1] 实现 90 度旋转
        return torch.cat([-x2, x1], dim=-1)

    # --- 步骤 1: 调整 cos/sin 的形状以支持广播 ---
    # 输入: q/k 形状为 [Batch, Seq_Len, Head, Dim]
    #       cos/sin 形状为 [Seq_Len, Dim]
    # 目标: 使 cos/sin 能够广播到 q/k 的所有维度
    #
    # 在 unsqueeze_dim=1 处插入维度:
    #   [Seq_Len, Dim] -> [Seq_Len, 1, Dim]
    # 广播后自动扩展为:
    #   [Seq_Len, 1, Dim] -> [Batch, Seq_Len, Head, Dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # --- 步骤 2: 应用旋转变换 ---
    # RoPE 的核心公式 (复数形式):
    #   q_rot = q * e^(i*theta) = q * (cos(theta) + i*sin(theta))
    #
    # 在实数域展开为:
    #   q_rot = q * cos(theta) + (q 旋转 90°) * sin(theta)
    #         = q * cos + rotate_half(q) * sin
    #
    # 这样每个位置的向量都会根据其位置索引进行不同角度的旋转,
    # 从而将位置信息隐式地编码到向量表示中
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def repeat_kv(
    x:torch.Tensor,
    n_rep:int
)->torch.Tensor:
    """
    重复键值(K/V)张量以支持分组查询注意力 (Grouped Query Attention, GQA)。

    在 GQA 中,K 和 V 的头数少于 Q 的头数,需要将每个 K/V 头复制多次以匹配 Q 的头数。
    例如:
    - 如果 Q 有 32 个头,K/V 有 8 个头,则 n_rep = 32 / 8 = 4
    - 每个 K/V 头会被复制 4 次,使得总头数从 8 变为 32

    这种设计的优势:
    - 减少 K/V 缓存的内存占用 (KV Cache)
    - 保持模型表达能力的同时降低计算成本
    - 在长序列生成时显著节省显存

    参数:
        x: 输入的 K 或 V 张量,形状 (Batch, Seq_Len, num_key_value_heads, head_dim)
        n_rep: 每个 K/V 头需要重复的次数,等于 num_attention_heads / num_key_value_heads

    返回:
        重复后的张量,形状 (Batch, Seq_Len, num_key_value_heads * n_rep, head_dim)
        其中 num_key_value_heads * n_rep = num_attention_heads
    """
    # 解构输入张量的形状
    bs, slen, num_key_value_heads, head_dim = x.shape

    # 优化: 如果不需要重复 (标准多头注意力 MHA),直接返回原张量
    if n_rep == 1:
        return x

    # --- 重复操作的三个步骤 ---
    # 步骤 1: 在第 3 维插入新维度 (None 等价于 unsqueeze)
    #   形状变化: [bs, slen, num_kv_heads, head_dim]
    #         -> [bs, slen, num_kv_heads, 1, head_dim]
    #
    # 步骤 2: 在新维度上扩展 n_rep 次 (expand 不复制数据,仅改变视图)
    #   形状变化: [bs, slen, num_kv_heads, 1, head_dim]
    #         -> [bs, slen, num_kv_heads, n_rep, head_dim]
    #
    # 步骤 3: 重塑为目标形状 (reshape 会触发实际的内存复制)
    #   形状变化: [bs, slen, num_kv_heads, n_rep, head_dim]
    #         -> [bs, slen, num_kv_heads * n_rep, head_dim]
    #
    # 效果: 每个 K/V 头被连续复制 n_rep 次
    # 例如: [h0, h1] 重复 3 次 -> [h0, h0, h0, h1, h1, h1]
    return (x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
# ==========================================
# 4. 注意力机制
# ==========================================


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        # --- 1. 确定 KV 头数 (GQA/MQA 核心逻辑) ---
        # 逻辑：决定 K 和 V 矩阵的维度。
        # - num_key_value_heads == num_attention_heads : 标准多头注意力 (MHA)
        # - num_key_value_heads < num_attention_heads  : 分组查询注意力 (GQA)
        # - num_key_value_heads == 1                   : 多查询注意力 (MQA，GQA 的特例)
        # GQA 能显著降低 KV Cache 显存占用，并提升推理吞吐量。
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        # --- 2. 维度合法性检查 ---
        # Q 的头数必须是 KV 头数的整数倍，以保证每个 KV 头能对应固定数量的 Q 头（构成一个组）。
        # 例：32 Q-heads / 8 KV-heads = 4 (每组 4 个 Q 共享 1 个 KV)
        if args.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({args.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        # --- 3. 记录核心维度参数 ---
        self.n_local_heads = args.num_attention_heads      # Q 的头数 (Query Heads)
        self.n_local_kv_heads = self.num_key_value_heads   # K/V 的头数 (Key/Value Heads)
        
        # [关键] 计算重复/广播倍率 (Repetition Factor)
        # 含义：每个 KV 头在计算 Attention Score 时需要匹配多少个 Q 头。
        # 作用：在 forward 阶段，KV 需要在维度上扩展 (expand/repeat) n_rep 倍以对齐 Q 的形状。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        
        # 计算单个头的维度 (Head Dimension)，例如: 4096 / 32 = 128 维
        # 注意：通常 Q, K, V 的 head_dim 必须一致。
        self.head_dim = args.hidden_size // args.num_attention_heads

        # --- 4. 定义投影层 (Projections) ---
        # Q 投影：全参数量。输入 hidden_size -> 输出 (Q头数 * 头维)
        # bias=False 是 Llama 类模型的常见设置，通常为了适配 RoPE 位置编码。
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        
        # K, V 投影：压缩参数 (GQA 特性)
        # 输入 hidden_size -> 输出 (KV头数 * 头维)
        # 当 n_rep > 1 时，这里的参数矩阵显著小于 q_proj，从而减少模型总参数量。
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False
        )
        
        # 输出投影：将所有头的结果拼接 (Concat) 后，映射回 hidden_size
        # 这里的输入维度与 Q 投影的输出维度一致（因为 Attention 输出形状跟随 Q）。
        self.o_proj = nn.Linear(
            self.n_local_heads * self.head_dim, args.hidden_size, bias=False
        )

        # --- 5. Dropout 与 加速开关 ---
        # Attention Matrix 的 Dropout：作用于 Softmax(QK^T) 之后，V 乘法之前。
        self.attn_dropout = nn.Dropout(args.dropout)
        
        # 残差路径的 Dropout：作用于 o_proj 之后，Residual Add 之前。
        self.resid_dropout = nn.Dropout(args.dropout)
        
        # 保存 float 类型的 dropout 概率，供 SDPA 函数式接口直接使用
        self.dropout = args.dropout
        
        # 检查是否启用 PyTorch 原生融合算子 (SDPA - Scaled Dot Product Attention)
        # 只要 PyTorch 版本 > 2.0 且 args 开启，即可使用。
        # 注意：SDPA 会根据硬件自动选择 Flash Attention (v2)、Memory Efficient Attention 或 Math 实现。
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )
        
    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_K_V: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,       
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 计算 Q, K, V 矩阵
        bsz,seq_len,_ = x.shape
        xq,xk,xv = self.q_proj(x),self.k_proj(x),self.v_proj(x)
        # 把输入拆分成多个头
        xq = xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        xk = xk.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        xv = xv.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        # 应用 RoPE 位置编码
        cos , sin = position_embeddings
        xq,xk = apply_rotary_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
        
        # KV 缓存机制：拼接历史缓存与当前计算的 key/value
        if past_K_V is not None:
            past_k,past_v = past_K_V
            # 拼接历史 key 缓存：
            # past_k [bsz, past_seq_len, n_local_kv_heads, head_dim] + xk [bsz, seq_len, n_local_kv_heads, head_dim]
            # 结果 xk 维度：
            # [bsz, past_seq_len + seq_len, n_local_kv_heads, head_dim]
            xk = torch.cat([past_k,xk],dim=1)
            # 拼接历史 value 缓存：
            # past_v [bsz, past_seq_len, n_local_kv_heads, head_dim] + xv [bsz, seq_len, n_local_kv_heads, head_dim]
            # 结果 xv 维度：[bsz, past_seq_len + seq_len, n_local_kv_heads, head_dim]
            xv = torch.cat([past_v,xv],dim=1)
        
        
        past_key_value = (xk,xv) if use_cache else None
        
        xq,xk,xv = (
            xq.transpose(1,2),  # [bsz, n_heads, seq_len, head_dim]
            repeat_kv(xk,self.n_rep).transpose(1,2),  
            repeat_kv(xv,self.n_rep).transpose(1,2)   # (B, Heads, Seq, Head_Dim)
        )
        
        if self.flash:
            output = F.scaled_dot_product_attention(
                xq, 
                xk, 
                xv,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # 手动实现 Scaled Dot-Product Attention
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 原代码：
            # causal_mask = torch.triu(
            #     torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
            #     diagonal=-1,
            # )

            # 修复：diagonal 应该为 1（保留对角线及以下），而非 -1
            # torch.triu(diagonal=1) 会将对角线上方的元素设为 -inf，实现因果掩码
            kv_seq_len = xk.shape[2]  # KV 的序列长度（可能因为 KV cache 而大于 seq_len）
            causal_mask = torch.triu(
                torch.full((seq_len, kv_seq_len), float("-inf"), device=scores.device, dtype=scores.dtype),
                diagonal=1,  # 修复：从 -1 改为 1
            )

            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
            
            
        # bsz,seq_len,n_heads,head_dim
        output=output.transpose(1,2).reshape(bsz,seq_len,-1)
        output=self.resid_dropout(self.o_proj(output))



        return output,past_key_value


class FeedForward(nn.Module):
    """
    前馈神经网络层 (Feed-Forward Network)
    使用 SwiGLU 激活函数：gate(x) * up(x)
    """
    # 初始化
    # 升维度
    # 降维
    # 门控
    # dropout
    # 激活函数
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        # 原代码：
        # if args.intermediate_size is None:
        #     intermediate_size = int(args.hidden_size * 8 / 3)
        #     args.intermediate_size = 64*((intermediate_size +64-1)//64)
        #
        #     self.up_proj = nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        #     ...

        # 修复：缩进错误，投影层应该在 if 外部定义，确保无论 intermediate_size 是否为 None 都能初始化
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)  # 默认8/3倍升维
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # 向上取64的倍数

        # 确保 intermediate_size 已设置
        assert args.intermediate_size is not None, "intermediate_size must be set"

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)  # dropout层
        self.act_fn = ACT2FN[args.hidden_act]  # 激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：SwiGLU 激活
        公式：dropout(down(act(up(x)) * gate(x)))
        """
        return self.dropout(
            self.down_proj(
                self.act_fn(self.up_proj(x)) * self.gate_proj(x)
            )
        )

class MokioMindBlock(nn.Module):
    def __init__(self, layer_id:int ,config:MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)
        
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)
    
    def forward(self,
                hidden_states:torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] =None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None,
                )->Tuple[torch.Tensor,Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Transformer Block 前向传播
        使用 Pre-Norm 架构：Norm -> Attention/FFN -> Residual
        """
        # === Attention 子层 ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            position_embeddings,
            past_K_V=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        
        hidden_states = residual + hidden_states

        # === FFN 子层 ===
        # 原代码：
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))

        # 修复：重复调用了 post_attention_layernorm，应该只调用一次
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = residual + self.mlp(hidden_states)  # 修复：移除重复的 layernorm 调用
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
    
    
    
class MokioMindModel(nn.Module):
    def __init__(self, config:MokioMindConfig):
        super().__init__()
        self.vocab_size , self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers
        )
        
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size)
        
        self.droppout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList(
            [MokioMindBlock(i,config) for i in range(self.num_hidden_layers)]
        )
        
        self.norm = RMSNorm(config.hidden_size , eps = config.rms_norm_eps)
        
        freqs_cos , freqs_sin = precompute_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_base,
            rope_scaling=config.rope_scaling,
        )
        self.freqs_cos: torch.Tensor 
        self.freqs_sin: torch.Tensor
        self.register_buffer("freqs_cos",freqs_cos,persistent=False)
        self.register_buffer("freqs_sin",freqs_sin,persistent=False)
        
    def forward(
        self,
        input_ids:Optional[torch.Tensor]=None,
        attention_mask:Optional[torch.Tensor]=None,
        past_key_values:Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]=None,  # 修复类型：使用 List 而非 Tuple
        use_cache:bool = False,
        **kwargs  #其他类的定义由这个接收
    ):

        if input_ids is None:
            raise ValueError("input_ids cannot be None for forward pass.")


        batch_size , seq_len = input_ids.shape

        # 原代码：
        # if hasattr(past_key_values,'layers'):
        #     past_key_values = None

        # 修复：添加类型检查，确保 past_key_values 是正确的类型
        if past_key_values is not None and hasattr(past_key_values, 'layers'):
            past_key_values = None

        # 原代码：
        # if past_key_values is None:
        #     past_key_values = [None] * len(self.layers)

        # 修复：使用类型安全的方式初始化
        past_key_values_list: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        if past_key_values is None:
            # 显式构造一个长度为层数的列表，初始全为 None
            past_key_values_list = [None for _ in range(len(self.layers))]
        else:
            past_key_values_list = past_key_values


        start_pos = 0
        # 原代码：
        # if past_key_values[0] is not None:
        #      start_pos = past_key_values[0][0].shape[2]

        # 修复：添加类型守卫，确保 past_key_values_list 不为空且第一个元素存在
        if len(past_key_values_list) > 0 and past_key_values_list[0] is not None:
             # past_key_values_list[0][0] 是 Key 矩阵: [Batch, Heads, SeqLen, Dim]
             start_pos = past_key_values_list[0][0].shape[2] # 注意维度索引通常是 2 (seq_len)

        hidden_states = self.droppout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos+seq_len],
            self.freqs_sin[start_pos:start_pos+seq_len],
        )

        presents: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []  # 添加类型注解


        # 原代码：
        # for layer_idx , (layer , past_key_values) in enumerate(
        #     zip(self.layers , past_key_values)
        # ):

        # 修复：重命名循环变量避免覆盖外部变量，使用类型安全的列表
        for layer_idx , (layer , past_kv) in enumerate(
            zip(self.layers , past_key_values_list)
        ):
            hidden_states , present = layer(
                hidden_states,
                position_embeddings,
                past_key_value = past_kv,  # 修复：使用正确的参数名
                use_cache = use_cache,
                attention_mask = attention_mask,
            )

            presents.append(present)


        hidden_states = self.norm(hidden_states)

        return hidden_states , presents
    
    
        
        
        

        
        
        
        
    
    
    
    
    
    
class MokioMindForCausalLM(PreTrainedModel,GenerationMixin): #hug定义的标准类
    config_class = MokioMindConfig
    
    def __init__(self,config:MokioMindConfig):
        self.config = config
        
        super().__init__(config)
        
        self.model = MokioMindModel(config)
        
        
        
    
    
    
    
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# class Attention(nn.Module):
#     def __init__(self, args: MokioMindConfig):
#         super().__init__()

#         # --- 1. 初始化 KV 头数配置 (GQA 核心逻辑) ---
#         # 逻辑：如果配置中未指定 num_key_value_heads (即为 None)，则默认它等于 num_attention_heads。
#         # 效果：
#         #   - 如果相等：退化为标准的 Multi-Head Attention (MHA)。
#         #   - 如果不等（通常更小）：启用 Grouped Query Attention (GQA)。
#         self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        
#         # --- 2. 维度检查 [优化] ---
#         # 检查 1: 隐藏层维度必须能被 Q 头数整除，算出 head_dim
#         assert args.hidden_size % args.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
#         # 检查 2: Q 头数必须能被 KV 头数整除 (这是 GQA 分组的前提)
#         assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
       
#         # --- 3. 核心参数设置 ---
#         # Query (Q) 的头数
#         self.n_local_heads = args.num_attention_heads
        
#         # Key/Value (K/V) 的头数
#         self.n_local_kv_heads = self.num_key_value_heads
        
#         # 【关键点】计算重复次数 (Repetition Factor)
#         # 含义：每个 KV 头对应多少个 Q 头。
#         # 作用：在 forward 阶段，KV 需要在第 2 维度上复制 n_rep 次，以对齐 Q 的形状进行计算。
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
        
#         # 计算每个头的维度 (head_dim)
#         self.head_dim = args.hidden_size // args.num_attention_heads
        
#         # --- 4. 定义线性投影层 (Linear Layers) ---
#         # Q 的投影层：输出维度 = 总头数 * 头维度
#         self.q_proj = nn.Linear(args.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        
#         # K, V 的投影层：输出维度 = KV头数 * 头维度
#         # [GQA优势]：当 n_local_kv_heads < n_local_heads 时，这里的参数量大幅减少。
#         self.k_proj = nn.Linear(args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
#         self.v_proj = nn.Linear(args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        
#         # 输出投影层
#         self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False) 
        
#         # --- 5. Dropout 设置 [修复 Bug] ---
#         # 原代码 bug：self.dropout 先被赋值为 nn.Dropout 对象，下一行又被赋值为 float 数值。
#         # 修复：将概率值存为不同的变量名 (如 dropout_p)，避免覆盖 nn.Module。
#         self.dropout = nn.Dropout(args.dropout)       # 这是一个 Layer (nn.Module)
#         self.resid_dropout = nn.Dropout(args.dropout) # 这是一个 Layer
#         self.dropout_p = args.dropout                 # [修复] 这是一个 float 数值，用于 functional 调用
        
#         # --- 6. Flash Attention 配置 [修复逻辑] ---
#         # 原代码 bug：末尾多了一个 `and args`，这在 Python 中总是 True (对象本身)，没有意义且容易引起误解。
#         # 修复：移除末尾多余部分。
#         self.flash_attention = (
#             args.flash_attention and 
#             hasattr(torch.nn.functional, "scaled_dot_product_attention")
#         )
