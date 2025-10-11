# 模型部署指南

提供多种部署方案，帮助您在不同环境中高效部署 MiniCPM-o 模型。

📖 [English Version](./README.md) | [返回主页](../)

## 部署框架对比

| 框架 | 性能 | 易用性 | 扩展性 | 硬件要求 | 适用场景 |
|------|------|--------|--------|---------|---------|
| [**vLLM**](./vllm/) | 高 | 中 | 高 | GPU | 大规模生产服务 |
| [**SGLang**](./sglang/) | 高 | 中 | 高 | GPU | 结构化生成任务 |
| [**Ollama**](./ollama/) | 中 | 极高 | 中 | CPU/GPU | 个人使用、快速原型 |
| [**Llama.cpp**](./llama.cpp/) | 中 | 高 | 中 | CPU | 边缘设备、轻量部署 |

## 框架详解

### [vLLM](./vllm/) (Very Large Language Model)
- 高吞吐量推理引擎，支持PagedAttention内存管理
- 动态批处理支持，OpenAI兼容API
- 适合生产环境API服务、大规模批量推理
- 推荐硬件：8G显存以上的GPU

### [SGLang](./sglang/) (Structured Generation Language)
- 结构化生成优化，高效的KV缓存管理
- 支持复杂控制流和函数调用优化
- 适合复杂推理链、结构化文本生成
- 推荐硬件：8G显存以上的GPU

### [Ollama](./ollama/)
- 一键式模型管理，简单的命令行界面
- 自动量化支持，REST API接口
- 适合个人开发环境、研究和原型验证
- 硬件要求：8GB+ RAM，支持CPU/GPU

### [Llama.cpp](./llama.cpp/)
- 纯C++实现，CPU优化推理
- 多种量化支持，轻量级部署
- 适合移动设备、边缘计算设备
- 硬件要求：4GB+ RAM，各种CPU架构

## 选择建议

- **生产环境 (高并发)**: vLLM - 最高性能、最佳扩展性
- **复杂推理任务**: SGLang - 结构化生成、函数调用优化
- **个人开发**: Ollama - 简单易用、快速上手
- **边缘部署**: Llama.cpp - 轻量级、低功耗 