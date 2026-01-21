#!/usr/bin/env python3
"""
测试 HuggingFace Token 是否配置正确
"""

import os
import sys

def test_hf_token():
    """测试 HuggingFace token 配置"""
    print("=" * 60)
    print("HuggingFace Token 配置测试")
    print("=" * 60)
    print()
    
    # 方法 1: 检查环境变量
    print("1. 检查环境变量:")
    token_env = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token_env:
        print(f"   ✓ 找到环境变量 token: {token_env[:10]}...{token_env[-4:]}")
    else:
        print("   ✗ 未找到环境变量 token")
    print()
    
    # 方法 2: 检查缓存文件
    print("2. 检查缓存文件:")
    cache_token_path = os.path.expanduser('~/.cache/huggingface/token')
    if os.path.exists(cache_token_path):
        try:
            with open(cache_token_path, 'r') as f:
                token_cache = f.read().strip()
                if token_cache:
                    print(f"   ✓ 找到缓存 token: {token_cache[:10]}...{token_cache[-4:]}")
                else:
                    print("   ✗ 缓存文件为空")
        except Exception as e:
            print(f"   ✗ 读取缓存文件失败: {e}")
    else:
        print("   ✗ 缓存文件不存在")
    print()
    
    # 方法 3: 验证 token
    print("3. 验证 token:")
    try:
        from huggingface_hub import HfApi
        
        # 优先使用环境变量，否则使用缓存
        token = token_env
        if not token and os.path.exists(cache_token_path):
            with open(cache_token_path, 'r') as f:
                token = f.read().strip()
        
        if token:
            api = HfApi(token=token)
            user_info = api.whoami()
            print(f"   ✓ Token 有效!")
            print(f"   ✓ 用户: {user_info.get('name', 'Unknown')}")
            print(f"   ✓ 类型: {user_info.get('type', 'Unknown')}")
            return True
        else:
            print("   ✗ 未找到有效的 token")
            return False
    except Exception as e:
        print(f"   ✗ Token 验证失败: {e}")
        return False

def test_model_access():
    """测试模型访问权限"""
    print()
    print("=" * 60)
    print("模型访问权限测试")
    print("=" * 60)
    print()
    
    models_to_test = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "KAKA22/CodeRM-8B"
    ]
    
    try:
        from huggingface_hub import HfApi
        
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        if not token:
            cache_token_path = os.path.expanduser('~/.cache/huggingface/token')
            if os.path.exists(cache_token_path):
                with open(cache_token_path, 'r') as f:
                    token = f.read().strip()
        
        api = HfApi(token=token) if token else HfApi()
        
        for model_id in models_to_test:
            print(f"测试模型: {model_id}")
            try:
                # 尝试获取模型信息
                model_info = api.model_info(model_id)
                print(f"   ✓ 可以访问")
                print(f"   - 模型 ID: {model_info.id}")
                print(f"   - 作者: {model_info.author}")
            except Exception as e:
                print(f"   ✗ 无法访问: {e}")
            print()
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == '__main__':
    # 测试 token
    token_ok = test_hf_token()
    
    # 如果 token 有效，测试模型访问
    if token_ok:
        test_model_access()
    else:
        print()
        print("=" * 60)
        print("建议:")
        print("=" * 60)
        print("1. 设置环境变量: export HF_TOKEN='your_token_here'")
        print("2. 或运行: huggingface-cli login")
        print("3. 或运行: hf auth login")
        sys.exit(1)
