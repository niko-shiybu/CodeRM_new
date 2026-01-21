#!/bin/bash

# GPU 检查脚本
# 用于验证服务器 GPU 是否可用

echo "=========================================="
echo "GPU 环境检查"
echo "=========================================="
echo ""

# 1. 检查 nvidia-smi
echo "1. 检查 NVIDIA 驱动:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s: %s, %s MB total, %s MB free\n", $1, $2, $3, $4}'
    echo "   ✓ nvidia-smi 可用"
else
    echo "   ✗ nvidia-smi 未找到"
    echo "   请安装 NVIDIA 驱动"
    exit 1
fi
echo ""

# 2. 检查 CUDA
echo "2. 检查 CUDA:"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "   ✓ CUDA 版本: $CUDA_VERSION"
else
    echo "   ⚠ CUDA 编译器未找到（PyTorch 可能使用预编译版本）"
fi
echo ""

# 3. 检查 Python 环境
echo "3. 检查 Python 环境:"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "   ✓ $PYTHON_VERSION"
else
    echo "   ✗ Python 未找到"
    exit 1
fi
echo ""

# 4. 检查 PyTorch 和 CUDA
echo "4. 检查 PyTorch 和 GPU:"
python << 'PYTHON_SCRIPT'
import sys
try:
    import torch
    print(f"   ✓ PyTorch 版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✓ CUDA 可用")
        print(f"   ✓ GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"     显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("   ✗ CUDA 不可用")
        print("   请检查 PyTorch 是否安装了 CUDA 版本")
        sys.exit(1)
except ImportError:
    print("   ✗ PyTorch 未安装")
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# 5. 检查 vLLM
echo "5. 检查 vLLM:"
python << 'PYTHON_SCRIPT'
try:
    from vllm import LLM
    print("   ✓ vLLM 已安装")
except ImportError:
    print("   ✗ vLLM 未安装")
    print("   请运行: pip install vllm")
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# 6. 检查 GPU 显存
echo "6. 检查 GPU 显存（用于推理）:"
python << 'PYTHON_SCRIPT'
import torch

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1024**3
        free_mem = (props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3
        
        print(f"   GPU {i}:")
        print(f"     总显存: {total_mem:.2f} GB")
        print(f"     可用显存: {free_mem:.2f} GB")
        
        # 检查是否足够运行 Llama3.1-8B（需要约 16GB）
        if free_mem >= 16:
            print(f"     ✓ 足够运行 Llama3.1-8B")
        elif free_mem >= 12:
            print(f"     ⚠ 可能勉强运行（建议至少 16GB）")
        else:
            print(f"     ✗ 显存不足（需要至少 16GB）")
PYTHON_SCRIPT
echo ""

# 7. 测试 GPU 推理（可选，会下载模型）
echo "7. 快速 GPU 推理测试（可选）:"
read -p "   是否运行快速测试？这会下载模型并测试推理 [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   正在测试..."
    python << 'PYTHON_SCRIPT'
from vllm import LLM, SamplingParams
import sys

try:
    print("   正在加载模型（首次运行会下载）...")
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        max_model_len=512,  # 小长度用于快速测试
        gpu_memory_utilization=0.5  # 降低显存使用
    )
    print("   ✓ 模型加载成功")
    
    sampling_params = SamplingParams(max_tokens=10, temperature=0.8)
    outputs = llm.generate(["Hello"], sampling_params)
    print(f"   ✓ 推理成功: {outputs[0].outputs[0].text[:50]}")
    print("   ✓ GPU 推理正常工作！")
except Exception as e:
    print(f"   ✗ 测试失败: {e}")
    sys.exit(1)
PYTHON_SCRIPT
fi

echo ""
echo "=========================================="
echo "检查完成！"
echo "=========================================="
echo ""
echo "如果所有检查都通过，可以运行："
echo "  cd inference"
echo "  python inference_mp.py --config config_sol_llama3-8b.json"
