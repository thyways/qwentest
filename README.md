# Qwen2VL-TriForce: 基于推测解码的高效多模态生成系统

## 概述

Qwen2VL-TriForce 是针对 Qwen2-VL (千问视觉-语言模型) 的优化推理系统，通过结合以下技术加速多模态输入(图像/视频)的文本生成：

- **推测性执行**：双模型协同的token预测-验证机制
- **动态稀疏注意力**：层次化视觉token剪枝算法
![系统架构](https://example.com/path/to/architecture.png) *(请替换为实际架构图)*

## 主要特性

- 🚀 相比自回归解码 **提速3.x倍**
- 🖼️ 支持 **图像和视频多模态输入**
- 💾 提供三种高效缓存机制：
  - `FlashSimpleCache`：基础键值缓存
  - `RetrievalCache`：基于层次化稀疏的注意力感知缓存
  - `StreamingLLMEvictionCache`：流式场景优化缓存
- 🔍 通过严格验证机制 **保持生成质量**

## 安装指南

### 环境要求

```bash
conda create -n Qwen-TriForce python=3.9
conda activate Qwen-TriForce
pip install -r requirements.txt
pip install flash-attn --no-build-isolation # install flash-attn
```
### 快速开始

```bash
git clone https://github.com/yourusername/Qwen2VL-TriForce.git
cd Qwen2VL-TriForce
python Qwen_TF.py
