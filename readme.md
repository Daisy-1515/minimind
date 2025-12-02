# MiniMind - 极简 Transformer 实现

[](https://www.python.org/)
[](https://pytorch.org/)
[](https://huggingface.co/transformers/)
[](https://www.google.com/search?q=LICENSE)

## 📖 简介

MiniMind 是一个极简的 Transformer 语言模型实现。它包含了一系列现代 LLM 的核心技术组件，代码结构清晰，易于学习和扩展。

### ✨ 核心特性

  - **分组查询注意力 (GQA)**: 优化 KV 缓存，提升推理速度。
  - **旋转位置编码 (RoPE)**: 增强模型的位置感知能力，支持外推。
  - **RMS 归一化**: 替代 LayerNorm，提升训练稳定性。
  - **SwiGLU 激活函数**: 现代 Transformer 标配，优于 ReLU。
  - **Flash Attention**: 集成 PyTorch 2.0+ 的加速算子。
  - **YaRN 长上下文**: 支持上下文窗口扩展。
  - **混合专家模型 (MoE)**: (计划中) 支持稀疏混合专家架构。
  - **结构清晰**: 模块化设计，易于阅读。
  - **从零预训练**: 包含完整的预训练流程。

## 📂 目录结构

```
minimind/
├── model/                  # 模型核心代码
│   ├── model.py           # MokioMind 主模型文件
│   │   ├── MokioMindConfig      # 配置类
│   │   ├── Attention            # 注意力机制
│   │   ├── FeedForward          # SwiGLU 前馈网络
│   │   ├── MokioMindBlock       # Transformer 层
│   │   ├── MokioMindModel       # 模型主体
│   │   └── MokioMindForCausalLM # 因果语言模型头
│
├── method/                 # 核心算法实现
│   ├── rope.py            # 旋转位置编码 (RoPE)
│   │   ├── precompute_freqs_cis # 预计算频率
│   │   └── apply_rotary_pos_emb # 应用 RoPE
│   ├── rmsnorm.py         # RMS 归一化层
│   └── gqa.py             # 分组查询注意力逻辑    
│
├── dataset/                # 数据集处理
│   └── lm_dataset.py      # 语言模型数据加载器    
│
├── trainer/                # 训练脚本
│   ├── train_pre.py       # 预训练主循环  
│   └── trainer_utils.py   # 训练工具函数 
│
├── environment/            # 环境配置
│   └── environment.yml    # Conda 环境文件
│
├── main.py                 # 推理演示脚本
├── 开发日志.md              # 开发过程记录
├── 常见问题.md              # 常见 Bug 与修复
└── 路线图.md                # 项目未来规划
```

## ⚙️ 参数配置

### 模型参数

MiniMind 使用以下默认参数配置（可自定义）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 512 | 隐藏层维度 |
| `num_attention_heads` | 8 | 注意力头数 |
| `num_key_value_heads` | 2 | KV 头数 (GQA 配置)     |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `vocab_size` | 6400 | 词表大小 |
| `max_position_embeddings` | 32768 | 最大上下文长度 |
| `rope_base` | 1000000 | RoPE 基频 |
| `flash_attention` | True | 是否开启 Flash Attention |

### 核心技术详解

#### 1\. 分组查询注意力 (GQA)

GQA 是 Multi-Query Attention (MQA) 和 Multi-Head Attention (MHA) 的折中方案：

```
原始 MHA: 32 Q-heads, 32 KV-heads  (1:1)
GQA:      32 Q-heads, 8 KV-heads   (4x 压缩)
MQA:      32 Q-heads, 1 KV-head    (32x 压缩)
```

**优势**：

  - 大幅减少 KV Cache 显存占用：`(n_kv_heads / n_heads)`
  - 保持了接近 MHA 的性能，速度接近 MQA。

#### 2\. 旋转位置编码 (RoPE)

RoPE 通过旋转矩阵注入绝对位置信息：

```python
# 伪代码
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

**优势**：

  - 具有良好的相对位置性质。
  - 支持通过 YaRN 等方法进行长度外推。

#### 3\. RMS 归一化

相比 LayerNorm，RMSNorm 去除了均值中心化，计算更简便：

```python
RMS(x) = sqrt(mean(x^2) + eps)
output = weights * (x / RMS(x))
```

**优势**：

  - 计算开销更小。
  - 在 LLM 训练中更稳定。

#### 4\. SwiGLU 激活函数

使用门控线性单元变体：

```
输入 → up_proj → act_fn ──┐
                          ⊗ (element-wise) → down_proj → 输出
     → gate_proj ─────────┘
```

**优势**：

  - 相比标准 FFN 有更好的性能表现。
  - Llama 和 PaLM 等大模型均采用此结构。

## 🚀 快速开始

### 环境要求

  - Python 3.13+
  - PyTorch 2.9.0+
  - Transformers 4.57.1+
  - CUDA 11.8+ (如需 GPU 加速)

### 安装步骤

#### 方法 1: 使用 Conda (推荐)

```bash
# 克隆仓库
git clone https://github.com/yourusername/minimind.git
cd minimind

# 创建并激活 Conda 环境
conda env create -f environment/environment.yml
conda activate mokiomind-video
```

#### 方法 2: 使用 pip

```bash
# 克隆仓库
git clone https://github.com/yourusername/minimind.git
cd minimind

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install torch>=2.9.0 transformers>=4.57.1 numpy>=2.3.4 pandas>=2.3.3
```

### 推理示例

```python
from model.model import MokioMindConfig, MokioMindForCausalLM
import torch

# 初始化配置
config = MokioMindConfig(
    hidden_size=512,
    num_attention_heads=8,
    num_key_value_heads=2,  # GQA: 4倍压缩
    num_hidden_layers=8,
    vocab_size=6400,
    max_position_embeddings=32768,
    flash_attention=True,
)

# 实例化模型
model = MokioMindForCausalLM(config)

# 构造输入
input_ids = torch.randint(0, config.vocab_size, (1, 128))  # (batch, seq_len)

# 前向传播
outputs = model(input_ids)
logits = outputs.logits  # (batch, seq_len, vocab_size)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
```

### 使用 KV Cache 生成

```python
# 预填充阶段
outputs = model(input_ids, use_cache=True)
past_key_values = outputs.past_key_values

# 生成下一个 token
new_token_id = torch.tensor([[next_token_id]])
outputs = model(
    new_token_id,
    past_key_values=past_key_values,
    use_cache=True
)
```

## 📊 模型规格

### 参数规模表

| 规格 | Hidden Size | Layers | Heads (Q/KV) | 参数量 |
|------|-------------|--------|--------------|--------|
| Tiny | 512 | 8 | 8/2 | \~25M |
| Small | 768 | 12 | 12/3 | \~60M |
| Base | 1024 | 16 | 16/4 | \~120M |
| Large | 2048 | 24 | 32/8 | \~500M |

### GQA 显存节省对比

以 Llama-7B 结构为例 (Batch=32, Len=4096):

| 模式 | KV Heads | KV Cache (Batch=1, Seq=2048) | 节省 |
|------|---------|------------------------------|------|
| MHA | 32 | \~2.0 GB | 基准 |
| GQA-8 | 8 | \~0.5 GB | 75% 节省 |
| MQA | 1 | \~0.06 GB | 97% 节省 |

## 🛠️ 进阶功能

### Flash Attention 实现

自动检测环境并使用优化算子：

```python
if torch.cuda.is_available() and torch.version.cuda >= "11.8":
    # 使用 Flash Attention v2 (自动)
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
else:
    # 慢速兜底实现
    scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
    scores = softmax(scores + causal_mask)
    output = scores @ v
```

### YaRN 长文本扩展

通过修改配置启用 YaRN：

```python
config = MokioMindConfig(
    max_position_embeddings=32768,
    rope_scaling={
        "type": "yarn",
        "factor": 4,  # 4倍扩展
        "original_max_position_embeddings": 2048,
        "beta_fast": 32,
        "beta_slow": 1,
    }
)
```

### 混合精度 RMSNorm

为了数值稳定性，RMSNorm 内部强制使用 FP32：

```python
def forward(self, x):
    # 强制转换为 FP32 进行统计量计算
    x_float = x.float()
    normed = self._norm_(x_float)
    # 转换回原始精度
    return (normed * self.weights).type_as(x)
```
