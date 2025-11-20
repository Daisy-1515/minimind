
import torch
import torch.nn as nn

# 继承nn.Module 类
class RMSNorm(nn.Module):

#__init__ 初始化
    def __init__(self,dim:int,eps:float = 1e-5):
        """
        初始化 RMSNorm 层
        参数:
            dim: 归一化的特征维度
            eps: 防止除零的极小值，默认 1e-5
        """
        super().__init__()
        self.dim = dim  # 特征维度
        self.eps = eps  # 数值稳定性参数
        self.weights = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数，初始化为全1向量
# _norm_
    def _norm_(self,x):
        """
        计算 RMS 归一化
        参数:
            x: 输入张量，形状为 (..., dim)
        返回:
            归一化后的张量，形状与输入相同
        """
        # 计算均方根 (RMS)
        rms = x*torch.rsqrt(torch.mean(x.pow(2),dim=-1,keepdim=True) + self.eps)
        # 归一化
        return  rms

# forward 前向传播
    def forward(self,x):
        """
        前向传播函数
        参数:
            x: 输入张量，形状为 (..., dim)
        返回:
            归一化并缩放后的张量，形状与输入相同
        """
        # 计算归一化并缩放
        return self.weights*self._norm_(x.float()).type_as(x)