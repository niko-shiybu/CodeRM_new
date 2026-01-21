# Llama3.1-8B 完整实验流程（详细版）

## 📋 目录

1. [实验概述](#实验概述)
2. [前置条件检查](#前置条件检查)
3. [阶段 1: 服务器 - 生成 Solutions](#阶段-1-服务器---生成-solutions)
4. [阶段 2: 服务器 - 生成 Unit Tests](#阶段-2-服务器---生成-unit-tests)
5. [阶段 3: 数据传输](#阶段-3-数据传输)
6. [阶段 4: 本地 - Docker 执行](#阶段-4-本地---docker-执行)
7. [阶段 5: 计算 Table 2 指标](#阶段-5-计算-table-2-指标)
8. [故障排查](#故障排查)
9. [预期结果](#预期结果)

---

## 实验概述

### 实验目标
复现 Table 2 中 **Llama3.1-8B** 作为 Reward Model 的实验结果。

### 实验配置
- **Policy Model**: Llama3.1-8B-Instruct（生成代码解决方案）
- **Reward Model**: Llama3.1-8B-Instruct（生成单元测试）
- **Benchmark**: HumanEval+（164 个问题）
- **每个问题生成**: 100 个 solutions + 100 个 unit tests
- **总执行次数**: 164 × 100 × 100 = 1,640,000 次单元测试执行

### 计算指标
- Accuracy (Acc)
- F1 Score
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Line Coverage（行覆盖率，新增）

### 工作流分配
- **服务器（GPU）**: 模型推理（生成 solutions 和 unit tests）
- **本地（Mac + Docker）**: 执行单元测试（不需要 GPU）

---

## 前置条件检查

### 服务器端检查

#### 1.1 检查 GPU 环境

```bash
# 在服务器上
cd /path/to/CodeRM
conda activate coderm

# 运行 GPU 检查脚本
bash check_gpu.sh
```

**预期输出**：
```
✓ nvidia-smi 可用
✓ PyTorch 版本: 2.x.x
✓ CUDA 可用
✓ GPU 数量: 1
✓ GPU 0: NVIDIA A100 (或类似)
✓ vLLM 已安装
```

**如果检查失败**：
- 检查 NVIDIA 驱动：`nvidia-smi`
- 检查 PyTorch CUDA：`python -c "import torch; print(torch.cuda.is_available())"`
- 安装 vLLM：`pip install vllm==0.6.3.post1`

#### 1.2 检查 HuggingFace Token

```bash
# 在服务器上
python test_token.py
```

**预期输出**：
```
✓ Token 有效!
✓ 用户: your_username
✓ 可以访问 meta-llama/Meta-Llama-3.1-8B-Instruct
```

**如果失败**：
- 运行：`huggingface-cli login` 或 `hf auth login`
- 或设置环境变量：`export HF_TOKEN="your_token"`

#### 1.3 检查数据文件

```bash
# 检查 benchmark 数据
ls -lh data/benchmark/input_humaneval+_sol.jsonl
ls -lh data/benchmark/input_humaneval+_ut.jsonl

# 应该看到两个文件，每个约 164 行
```

**如果文件不存在**：
- 从项目仓库下载
- 或使用 `wget`/`curl` 下载

### 本地（Mac）检查

#### 2.1 检查 Docker

```bash
# 在本地 Mac 上
docker --version
docker ps
```

**预期输出**：
```
Docker version 24.x.x
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```

**如果失败**：
- 安装 Docker Desktop：https://www.docker.com/products/docker-desktop
- 启动 Docker Desktop 应用

#### 2.2 拉取 Docker 镜像

```bash
# 在本地
docker pull kaka0605/exec_unit_test:24.12.30

# 验证镜像
docker images | grep exec_unit_test
```

**预期输出**：
```
kaka0605/exec_unit_test   24.12.30    xxxxx    xxxxx    xxxx MB
```

#### 2.3 检查 Python 环境

```bash
# 在本地
cd /Users/fyc/Desktop/CodeRM
python3 --version

# 检查必要的 Python 包
python3 -c "import json, tqdm; print('OK')"
```

---

## 阶段 1: 服务器 - 生成 Solutions

### 步骤 1.1: 准备配置文件

确认配置文件存在且正确：

```bash
# 在服务器上
cd /path/to/CodeRM/inference
cat config_sol_llama3-8b.json
```

**配置文件内容**：
```json
{
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "dtype": "auto",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.8,
    "max_num_seqs": 512,
    "tensor_parallel_size": 1,
    "sampling_params": {
        "n": 100,
        "max_tokens": 2048,
        "top_p": 0.95,
        "temperature": 0.8,
        "stop": null
    },
    "num_gpus": 1,
    "output_dir": "../output/inference/llama3-8b_solutions",
    "messages_file": "../data/benchmark/input_humaneval+_sol.jsonl"
}
```

**关键参数说明**：
- `n: 100`: 每个问题生成 100 个解决方案
- `num_gpus: 1`: 使用 1 个 GPU
- `gpu_memory_utilization: 0.8`: 使用 80% 的 GPU 显存

### 步骤 1.2: 运行推理

```bash
# 在服务器上
cd /path/to/CodeRM/inference

# 运行推理（这会自动使用 GPU）
python inference_mp.py --config config_sol_llama3-8b.json
```

**执行过程**：
1. 代码会自动调用 `nvidia-smi` 检测可用 GPU
2. 分配 GPU（根据 `num_gpus: 1`）
3. 下载模型（首次运行，约 16GB，需要时间）
4. 加载模型到 GPU 显存
5. 开始生成 solutions（164 个问题 × 100 个 = 16,400 个生成任务）

**预计时间**：
- 模型下载：10-30 分钟（首次运行，取决于网络）
- 模型加载：1-2 分钟
- 生成 solutions：2-4 小时（取决于 GPU 性能）

**监控 GPU**（在另一个终端）：
```bash
watch -n 1 nvidia-smi
```

**预期看到**：
- GPU 显存使用：约 12-16GB
- GPU 利用率：接近 100%
- 温度：可能上升（正常）

**输出文件**：
```
output/inference/llama3-8b_solutions/output_gpu_0.jsonl
```

### 步骤 1.3: 合并输出文件

```bash
# 在服务器上
cd /path/to/CodeRM/preprocess

# 合并输出（mp_num = 使用的 GPU 数量，这里是 1）
python merge_output.py --mp_num 1 --input_dir ../output/inference/llama3-8b_solutions
```

**输出文件**：
```
output/inference/llama3-8b_solutions/merge_result.jsonl
```

**验证**：
```bash
# 检查文件大小（应该有几 GB）
ls -lh output/inference/llama3-8b_solutions/merge_result.jsonl

# 检查行数（应该是 164 行，每行一个问题的 100 个 responses）
wc -l output/inference/llama3-8b_solutions/merge_result.jsonl
```

### 步骤 1.4: 提取 Solutions

```bash
# 在服务器上
cd /path/to/CodeRM/preprocess

python extract_solution.py \
  --data_path ../output/inference/llama3-8b_solutions/merge_result.jsonl \
  --id_path ../data/benchmark/input_humaneval+_sol.jsonl \
  --output_path ../data/result/humaneval+/sol_llama3-8b_200.jsonl
```

**输出文件**：
```
data/result/humaneval+/sol_llama3-8b_200.jsonl
```

**验证**：
```bash
# 检查文件
head -n 1 data/result/humaneval+/sol_llama3-8b_200.jsonl | python3 -m json.tool

# 应该看到类似：
# {
#   "task_id": "HumanEval/0",
#   "solutions": [
#     "def has_close_elements(...)",
#     ...
#   ]
# }
```

**注意**：如果你已经有 `sol_llama3-8b_200_anno.jsonl`（标注文件），可以跳过生成步骤，直接使用现有文件。但为了完整复现，建议重新生成。

---

## 阶段 2: 服务器 - 生成 Unit Tests

### 步骤 2.1: 准备配置文件

```bash
# 在服务器上
cd /path/to/CodeRM/inference
cat config_ut_llama3-8b.json
```

**配置文件内容**：
```json
{
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "dtype": "auto",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.8,
    "max_num_seqs": 512,
    "tensor_parallel_size": 1,
    "sampling_params": {
        "n": 100,
        "max_tokens": 2048,
        "top_p": 0.95,
        "temperature": 0.8,
        "stop": null
    },
    "num_gpus": 1,
    "output_dir": "../output/inference/llama3-8b_unit_tests",
    "messages_file": "../data/benchmark/input_humaneval+_ut.jsonl"
}
```

### 步骤 2.2: 运行推理

```bash
# 在服务器上
cd /path/to/CodeRM/inference

# 运行推理
python inference_mp.py --config config_ut_llama3-8b.json
```

**执行过程**：
- 与生成 solutions 类似
- 164 个问题 × 100 个 = 16,400 个生成任务
- 预计时间：2-4 小时

**输出文件**：
```
output/inference/llama3-8b_unit_tests/output_gpu_0.jsonl
```

### 步骤 2.3: 合并输出

```bash
# 在服务器上
cd /path/to/CodeRM/preprocess

python merge_output.py --mp_num 1 --input_dir ../output/inference/llama3-8b_unit_tests
```

**输出文件**：
```
output/inference/llama3-8b_unit_tests/merge_result.jsonl
```

### 步骤 2.4: 提取 Unit Tests

```bash
# 在服务器上
cd /path/to/CodeRM/preprocess

python extract_unit_test.py \
  --input_path ../output/inference/llama3-8b_unit_tests/merge_result.jsonl \
  --id_path ../data/benchmark/input_humaneval+_ut.jsonl \
  --output_path ../data/result/humaneval+/ut_llama3-8b_100.jsonl
```

**输出文件**：
```
data/result/humaneval+/ut_llama3-8b_100.jsonl
```

**验证**：
```bash
# 检查文件
head -n 1 data/result/humaneval+/ut_llama3-8b_100.jsonl | python3 -m json.tool

# 应该看到类似：
# {
#   "task_id": "HumanEval/0",
#   "unit_tests": [
#     "import unittest\nclass Test...",
#     ...
#   ]
# }
```

---

## 阶段 3: 数据传输

### 步骤 3.1: 从服务器下载必要文件

**在本地 Mac 上执行**（替换 `user@server:/path/to/CodeRM` 为你的实际服务器地址）：

```bash
# 在本地
cd /Users/fyc/Desktop/CodeRM

# 创建必要的目录
mkdir -p data/result/humaneval+
mkdir -p data/benchmark
mkdir -p output/humaneval+/llama3-8b_sol_llama3-8b_ut/details

# 1. 下载 solutions（标注文件，包含正确/错误标签）
scp user@server:/path/to/CodeRM/data/result/humaneval+/sol_llama3-8b_200_anno.jsonl \
   ./data/result/humaneval+/

# 2. 下载 unit tests
scp user@server:/path/to/CodeRM/data/result/humaneval+/ut_llama3-8b_100.jsonl \
   ./data/result/humaneval+/

# 3. 下载 benchmark 数据（如果还没有）
scp user@server:/path/to/CodeRM/data/benchmark/input_humaneval+_sol.jsonl \
   ./data/benchmark/
scp user@server:/path/to/CodeRM/data/benchmark/input_humaneval+_ut.jsonl \
   ./data/benchmark/
```

**验证下载**：
```bash
# 检查文件是否存在
ls -lh data/result/humaneval+/sol_llama3-8b_200_anno.jsonl
ls -lh data/result/humaneval+/ut_llama3-8b_100.jsonl
ls -lh data/benchmark/input_humaneval+_sol.jsonl
ls -lh data/benchmark/input_humaneval+_ut.jsonl

# 检查文件大小（应该都是几 MB 到几十 MB）
```

**如果 scp 很慢**：
- 可以使用 `rsync`（支持断点续传）：
```bash
rsync -avz --progress user@server:/path/to/CodeRM/data/result/humaneval+/sol_llama3-8b_200_anno.jsonl \
   ./data/result/humaneval+/
```

---

## 阶段 4: 本地 - Docker 执行

### 步骤 4.1: 使用自动化脚本（推荐）

```bash
# 在本地 Mac 上
cd /Users/fyc/Desktop/CodeRM

# 运行自动化脚本
bash run_docker_local.sh
```

**脚本会自动**：
1. 检查 Docker 和镜像
2. 检查数据文件
3. 生成 `sol_ut.jsonl`（解决方案-单元测试组合文件）
4. 执行 Docker 容器
5. 保存结果文件
6. 清理临时文件

**预计时间**：2-6 小时（取决于 Mac 性能）

### 步骤 4.2: 手动执行（如果需要）

如果脚本有问题，可以手动执行：

#### 4.2.1 生成 sol_ut.jsonl

```bash
# 在本地
cd /Users/fyc/Desktop/CodeRM

# 创建输出目录
mkdir -p output/humaneval+/llama3-8b_sol_llama3-8b_ut/details

# 生成组合文件
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')
from evaluation.evaluate import save_sol_and_ut_comb

print("生成 sol_ut.jsonl...")
save_sol_and_ut_comb('humaneval+', 'llama3-8b', 'llama3-8b', 100, 100)
print("完成！")
PYTHON_SCRIPT
```

**输出文件**：
```
output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/sol_ut.jsonl
```

**验证**：
```bash
# 检查文件大小（应该很大，几 GB）
ls -lh output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/sol_ut.jsonl

# 检查行数（应该是 1,640,000 行）
wc -l output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/sol_ut.jsonl
```

#### 4.2.2 执行 Docker

```bash
# 在本地
cd /Users/fyc/Desktop/CodeRM

# 创建临时目录
TEMP_DIR=$(mktemp -d -t docker_write_XXXXXX)
chmod 777 "$TEMP_DIR"
echo "临时目录: $TEMP_DIR"

# 执行 Docker 容器
docker run -v "$(pwd):/data" kaka0605/exec_unit_test:24.12.30 \
    --input_path /data/output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/sol_ut.jsonl \
    --output_path /data/$TEMP_DIR/100_sol_100_ut_result.jsonl \
    --mp_num 8 \
    --chunk_size 1000 \
    --recover 0
```

**参数说明**：
- `--mp_num 8`: 使用 8 个进程并行执行
- `--chunk_size 1000`: 每次处理 1000 条
- `--recover 0`: 不恢复（从头开始）

**监控进度**：
Docker 会输出进度信息，显示已处理的数量。

**预计时间**：2-6 小时

#### 4.2.3 保存结果

```bash
# 移动结果文件
mv $TEMP_DIR/100_sol_100_ut_result.jsonl \
   output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/

# 清理
rm output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/sol_ut.jsonl
rmdir $TEMP_DIR
```

**输出文件**：
```
output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/100_sol_100_ut_result.jsonl
```

**验证**：
```bash
# 检查文件
head -n 1 output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/100_sol_100_ut_result.jsonl | python3 -m json.tool

# 应该看到类似：
# {
#   "task_id": "HumanEval/0",
#   "sol_id": 0,
#   "ut_id": 0,
#   "result": "pass"  // 或 "fail" 或 "error"
# }
```

---

## 阶段 5: 计算 Table 2 指标

### 步骤 5.1: 计算指标（不计算覆盖率，快速）

```bash
# 在本地（或服务器，如果结果文件在服务器上）
cd /Users/fyc/Desktop/CodeRM

python evaluation/calculate_table2_metrics.py \
  --benchmark humaneval+ \
  --sol_model llama3-8b \
  --ut_model llama3-8b \
  --sol_num 100 \
  --ut_num 100 \
  --mode both \
  --output_dir output/table2_results
```

**参数说明**：
- `--mode both`: 计算 Individual 和 Multiple 两种模式的指标
- `--output_dir`: 结果保存目录

**预计时间**：10-30 分钟

**输出文件**：
```
output/table2_results/humaneval+_llama3-8b_llama3-8b.json
```

**输出内容**：
```json
{
  "individual": {
    "accuracy": 60.02,
    "f1": 44.97,
    "far": 13.66,
    "frr": 46.13,
    "line_coverage": 0.0
  },
  "multiple": {
    "accuracy": 74.21,
    "f1": 74.35,
    "far": 20.44,
    "frr": 30.55,
    "line_coverage": 0.0
  }
}
```

### 步骤 5.2: 计算指标（包含行覆盖率，慢）

如果需要计算行覆盖率：

```bash
python evaluation/calculate_table2_metrics.py \
  --benchmark humaneval+ \
  --sol_model llama3-8b \
  --ut_model llama3-8b \
  --sol_num 100 \
  --ut_num 100 \
  --mode both \
  --output_dir output/table2_results \
  --calculate_coverage
```

**注意**：行覆盖率计算很慢，会对每个 solution 的前几个 unit test 采样计算。

### 步骤 5.3: 生成汇总表格

```bash
# 在本地
python evaluation/generate_table2_summary.py \
  --results_dir output/table2_results \
  --benchmark humaneval+ \
  --sol_model llama3-8b
```

**输出文件**：
```
output/table2_results/table2_summary_humaneval+.md
```

**输出内容**（Markdown 表格）：
```markdown
# Table 2: Quality of Unit Tests

## Quality of Individual Unit Tests

| Model | Acc (↑) | F1 (↑) | FAR (↓) | FRR (↓) | Line Coverage (↑) |
|-------|---------|--------|---------|---------|-------------------|
| Llama3.1-8B | 60.02 | 44.97 | 13.66 | 46.13 | 0.00 |

## Quality of Multiple Unit Tests

| Model | Acc (↑) | F1 (↑) | FAR (↓) | FRR (↓) | Line Coverage (↑) |
|-------|---------|--------|---------|---------|-------------------|
| Llama3.1-8B | 74.21 | 74.35 | 20.44 | 30.55 | 0.00 |
```

---

## 故障排查

### 问题 1: GPU 检测失败

**错误**：
```
AssertionError: len(free_gpus) >= config['num_gpus']
```

**解决**：
1. 检查 GPU：`nvidia-smi`
2. 降低阈值（修改 `inference_mp.py` 第 158 行）：
   ```python
   free_gpus = get_free_gpus(threshold=10000)  # 改为 10GB
   ```
3. 手动指定 GPU：
   ```python
   free_gpus = [0]  # 使用 GPU 0
   ```

### 问题 2: 显存不足（OOM）

**错误**：
```
CUDA out of memory
```

**解决**：
1. 降低 `gpu_memory_utilization`（配置文件）：
   ```json
   "gpu_memory_utilization": 0.7
   ```
2. 减少 `max_num_seqs`：
   ```json
   "max_num_seqs": 256
   ```
3. 减少 `max_model_len`：
   ```json
   "max_model_len": 2048
   ```

### 问题 3: Docker 执行失败

**错误**：
```
permission denied
```

**解决**：
1. 检查 Docker 是否运行：`docker ps`
2. 检查镜像：`docker images | grep exec_unit_test`
3. 重新拉取镜像：`docker pull kaka0605/exec_unit_test:24.12.30`

### 问题 4: 数据文件缺失

**错误**：
```
FileNotFoundError: ...
```

**解决**：
1. 检查文件路径是否正确
2. 确认文件已从服务器下载
3. 检查文件权限：`ls -lh data/result/humaneval+/`

### 问题 5: 指标计算错误

**错误**：
```
KeyError: 'plus_status'
```

**解决**：
1. 确认使用 `sol_llama3-8b_200_anno.jsonl`（标注文件）
2. 检查文件格式是否正确
3. 确认 benchmark 名称是 `humaneval+`（不是 `humaneval`）

---

## 预期结果

根据论文 Table 2，Llama3.1-8B 的预期结果：

### Individual Unit Tests
| 指标 | 预期值 |
|------|--------|
| Accuracy | 60.02 |
| F1 Score | 44.97 |
| FAR | 13.66 |
| FRR | 46.13 |

### Multiple Unit Tests
| 指标 | 预期值 |
|------|--------|
| Accuracy | 74.21 |
| F1 Score | 74.35 |
| FAR | 20.44 |
| FRR | 30.55 |

**注意**：实际结果可能略有差异，因为：
- 随机性（temperature, seed）
- 模型版本差异
- 硬件差异

---

## 时间估算

| 阶段 | 预计时间 |
|------|----------|
| 服务器 GPU 推理（Solutions） | 2-4 小时 |
| 服务器 GPU 推理（Unit Tests） | 2-4 小时 |
| 数据传输 | 5-30 分钟 |
| 本地 Docker 执行 | 2-6 小时 |
| 指标计算 | 10-30 分钟 |
| **总计** | **6-15 小时** |

---

## 检查清单

完成每个阶段后，检查以下项目：

### 阶段 1 完成
- [ ] `output/inference/llama3-8b_solutions/output_gpu_0.jsonl` 存在
- [ ] `output/inference/llama3-8b_solutions/merge_result.jsonl` 存在
- [ ] `data/result/humaneval+/sol_llama3-8b_200.jsonl` 存在

### 阶段 2 完成
- [ ] `output/inference/llama3-8b_unit_tests/output_gpu_0.jsonl` 存在
- [ ] `output/inference/llama3-8b_unit_tests/merge_result.jsonl` 存在
- [ ] `data/result/humaneval+/ut_llama3-8b_100.jsonl` 存在

### 阶段 3 完成
- [ ] 本地有 `data/result/humaneval+/sol_llama3-8b_200_anno.jsonl`
- [ ] 本地有 `data/result/humaneval+/ut_llama3-8b_100.jsonl`
- [ ] 本地有 `data/benchmark/input_humaneval+_sol.jsonl`
- [ ] 本地有 `data/benchmark/input_humaneval+_ut.jsonl`

### 阶段 4 完成
- [ ] `output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/100_sol_100_ut_result.jsonl` 存在
- [ ] 文件大小合理（几 GB）
- [ ] 文件包含 1,640,000 行结果

### 阶段 5 完成
- [ ] `output/table2_results/humaneval+_llama3-8b_llama3-8b.json` 存在
- [ ] `output/table2_results/table2_summary_humaneval+.md` 存在
- [ ] 指标值在合理范围内

---

## 下一步

完成 Llama3.1-8B 后，可以继续：

1. **Llama3.1-70B**：
   - 需要 4 张 GPU
   - 修改配置文件：`config_ut_llama3-70b.json`
   - 重复相同流程

2. **CodeRM-8B**：
   - 需要 1 张 GPU
   - 修改配置文件：`config_ut_coderm-8b.json`
   - 重复相同流程

3. **生成完整 Table 2**：
   ```bash
   python evaluation/generate_table2_summary.py \
     --results_dir output/table2_results \
     --benchmark humaneval+ \
     --sol_model llama3-8b
   ```

---

## 总结

这个流程涵盖了从 GPU 推理到指标计算的完整过程。关键点：

1. ✅ **服务器自动使用 GPU**：代码会自动检测和分配 GPU
2. ✅ **本地执行 Docker**：不需要 GPU，可以在 Mac 上运行
3. ✅ **数据通过 scp 传输**：简单可靠
4. ✅ **自动化脚本**：简化 Docker 执行流程

如果遇到问题，参考故障排查部分，或运行检查脚本诊断问题。

**祝实验顺利！** 🚀
