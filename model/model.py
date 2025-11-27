from ast import arg
from filecmp import BUFSIZE
import math
import re
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from typing import Optional, Tuple
from torch.nn import functional as F
from transformers.activations import ACT2FN

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

def repeat_kv(
    x:torch.Tensor,
    n_rep:int
)->torch.Tensor:
    """
    """
    bs,slen,num_key_value_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    
    return (x[:,:,:,None,:]
            .expand(bs,slen,num_key_value_heads,n_rep,head_dim)
            .reshape(bs,slen,num_key_value_heads*n_rep,head_dim))


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
        # 对于k和v，进行重复以匹配q的头数
        if past_K_V is not None:
            past_k,past_v = past_K_V
            xk = torch.cat([past_k,xk],dim=1)
            xv = torch.cat([past_v,xv],dim=1)
        
        past_key_value = (xk,xv) if use_cache else None
        
        xq,xk,xv = (
            
            xq.transpose(1,2),  # [bsz, n_heads, seq_len, head_dim]
            repeat_kv(xk,self.n_rep).transpose(1,2),  
            repeat_kv(xv,self.n_rep).transpose(1,2)   # (B, Heads, Seq, Head_Dim)
        )
        
        if self.flash:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=-1,
            )

            scores=scores+causal_mask.unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            output=scores@xv
        # bsz,seq_len,n_heads,head_dim
        output=output.transpose(1,2).reshape(bsz,seq_len,-1)
        output=self.resid_dropout(self.o_proj(output))



        return output,past_key_value


class FeedForward(nn.Module):
    
    #初始化
    #升维度
    #降维
    #门控
    #dropout
    #激活函数
    def __init__(self, args: MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3) #默认8/3倍升维
            args.intermediate_size = 64*((intermediate_size +64-1)//64  ) #向上取64的倍数
            
            self.up_proj = nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
            self.down_proj = nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
            self.gate_proj = nn.Linear(args.hidden_size,args.intermediate_size,bias=False) 
            self.dropout = nn.Dropout(args.dropout) #dropout层
            self.act_fn = ACT2FN[args.hidden_act]  #激活函数
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.dropout(
            self.down_proj(
                self.act_fn(self.up_proj(x))*self.gate_proj(x)
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
        
        
        
        
        
        
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value
    
    
    
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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
