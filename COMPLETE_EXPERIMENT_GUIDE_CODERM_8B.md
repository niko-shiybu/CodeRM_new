# CodeRM-8B 完整实验流程（详细版）

## 📋 概述

本指南详细说明如何使用 **CodeRM-8B** 作为 Reward Model 复现 Table 2 的实验。

### 实验配置
- **Policy Model**: Llama3.1-8B-Instruct（生成代码解决方案）
- **Reward Model**: CodeRM-8B（生成单元测试）
- **Benchmark**: HumanEval+（164 个问题）
- **每个问题生成**: 100 个 solutions + 100 个 unit tests

### 与 Llama3.1-8B 的区别
- ✅ **Policy Model 相同**：仍然使用 Llama3.1-8B 生成 solutions
- ✅ **Reward Model 不同**：使用 CodeRM-8B 生成 unit tests
- ✅ **GPU 需求相同**：1 张 GPU（~16GB 显存）
- ✅ **流程基本相同**：只需修改配置文件

---

## 🚀 快速开始

如果你已经完成了 Llama3.1-8B 的实验，**只需要修改步骤 2**（生成 Unit Tests）：

```bash
# 在服务器上
cd inference
python inference_mp.py --config config_ut_coderm-8b.json  # 只改这一行！
```

然后继续后续步骤，将 `llama3-8b` 替换为 `coderm-8b`。

---

## 📝 完整流程

### 阶段 1: 服务器 - 生成 Solutions（与 Llama3.1-8B 相同）

**注意**：如果你已经完成了 Llama3.1-8B 的实验，可以**跳过这一步**，直接使用已有的 solutions。

```bash
# 在服务器上
cd /path/to/CodeRM
conda activate coderm

# 1. 生成 solutions
cd inference
python inference_mp.py --config config_sol_llama3-8b.json

# 2. 合并输出
cd ../preprocess
python merge_output.py --mp_num 1 --input_dir ../output/inference/llama3-8b_solutions

# 3. 提取 solutions
python extract_solution.py \
  --data_path ../output/inference/llama3-8b_solutions/merge_result.jsonl \
  --id_path ../data/benchmark/input_humaneval+_sol.jsonl \
  --output_path ../data/result/humaneval+/sol_llama3-8b_200.jsonl
```

**输出文件**：
- `data/result/humaneval+/sol_llama3-8b_200_anno.jsonl`（标注文件，需要这个）

---

### 阶段 2: 服务器 - 生成 Unit Tests（CodeRM-8B）

#### 步骤 2.1: 检查配置文件

```bash
# 在服务器上
cd /path/to/CodeRM/inference
cat config_ut_coderm-8b.json
```

**配置文件内容**：
```json
{
    "model_path": "KAKA22/CodeRM-8B",
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
    "output_dir": "../output/inference/coderm-8b_unit_tests",
    "messages_file": "../data/benchmark/input_humaneval+_ut.jsonl"
}
```

**关键参数**：
- `model_path`: `KAKA22/CodeRM-8B`（CodeRM-8B 模型）
- `num_gpus`: 1（需要 1 张 GPU）
- `n: 100`: 每个问题生成 100 个 unit tests

#### 步骤 2.2: 运行推理

```bash
# 在服务器上
cd /path/to/CodeRM/inference

# 运行推理（使用 CodeRM-8B）
python inference_mp.py --config config_ut_coderm-8b.json
```

**执行过程**：
1. 自动检测 GPU
2. 下载 CodeRM-8B 模型（首次运行，约 16GB）
3. 加载模型到 GPU
4. 生成 unit tests（164 个问题 × 100 个 = 16,400 个生成任务）

**预计时间**：
- 模型下载：10-30 分钟（首次运行）
- 模型加载：1-2 分钟
- 生成 unit tests：2-4 小时

**监控 GPU**（在另一个终端）：
```bash
watch -n 1 nvidia-smi
```

**输出文件**：
```
output/inference/coderm-8b_unit_tests/output_gpu_0.jsonl
```

#### 步骤 2.3: 合并输出

```bash
# 在服务器上
cd /path/to/CodeRM/preprocess

python merge_output.py --mp_num 1 --input_dir ../output/inference/coderm-8b_unit_tests
```

**输出文件**：
```
output/inference/coderm-8b_unit_tests/merge_result.jsonl
```

**验证**：
```bash
ls -lh output/inference/coderm-8b_unit_tests/merge_result.jsonl
wc -l output/inference/coderm-8b_unit_tests/merge_result.jsonl  # 应该是 164 行
```

#### 步骤 2.4: 提取 Unit Tests

```bash
# 在服务器上
cd /path/to/CodeRM/preprocess

python extract_unit_test.py \
  --input_path ../output/inference/coderm-8b_unit_tests/merge_result.jsonl \
  --id_path ../data/benchmark/input_humaneval+_ut.jsonl \
  --output_path ../data/result/humaneval+/ut_coderm-8b_100.jsonl
```

**输出文件**：
```
data/result/humaneval+/ut_coderm-8b_100.jsonl
```

**验证**：
```bash
# 检查文件
head -n 1 data/result/humaneval+/ut_coderm-8b_100.jsonl | python3 -m json.tool
```

---

### 阶段 3: 数据传输

从服务器下载 CodeRM-8B 的 unit tests 文件：

```bash
# 在本地 Mac 上执行
cd /Users/fyc/Desktop/CodeRM

# 下载 CodeRM-8B 的 unit tests（关键！）
scp user@server:/path/to/CodeRM/data/result/humaneval+/ut_coderm-8b_100.jsonl \
   ./data/result/humaneval+/

# 如果还没有 solutions 文件，也下载
scp user@server:/path/to/CodeRM/data/result/humaneval+/sol_llama3-8b_200_anno.jsonl \
   ./data/result/humaneval+/

# 确保 benchmark 数据存在
scp user@server:/path/to/CodeRM/data/benchmark/input_humaneval+_sol.jsonl \
   ./data/benchmark/
scp user@server:/path/to/CodeRM/data/benchmark/input_humaneval+_ut.jsonl \
   ./data/benchmark/
```

**验证**：
```bash
ls -lh data/result/humaneval+/ut_coderm-8b_100.jsonl
ls -lh data/result/humaneval+/sol_llama3-8b_200_anno.jsonl
```

---

### 阶段 4: 本地 - Docker 执行

#### 方法 A: 修改自动化脚本

编辑 `run_docker_local.sh`，修改以下变量：

```bash
UT_MODEL="coderm-8b"  # 改为 coderm-8b
```

然后运行：
```bash
bash run_docker_local.sh
```

#### 方法 B: 手动执行

```bash
# 在本地 Mac 上
cd /Users/fyc/Desktop/CodeRM

# 创建输出目录
mkdir -p output/humaneval+/llama3-8b_sol_coderm-8b_ut/details

# 生成 sol_ut.jsonl（注意：sol_model 是 llama3-8b，ut_model 是 coderm-8b）
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')
from evaluation.evaluate import save_sol_and_ut_comb

save_sol_and_ut_comb('humaneval+', 'llama3-8b', 'coderm-8b', 100, 100)
print("完成！")
PYTHON_SCRIPT

# 执行 Docker
TEMP_DIR=$(mktemp -d -t docker_write_XXXXXX)
chmod 777 "$TEMP_DIR"

docker run -v "$(pwd):/data" kaka0605/exec_unit_test:24.12.30 \
    --input_path /data/output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/sol_ut.jsonl \
    --output_path /data/$TEMP_DIR/100_sol_100_ut_result.jsonl \
    --mp_num 8 \
    --chunk_size 1000 \
    --recover 0

# 保存结果
mv $TEMP_DIR/100_sol_100_ut_result.jsonl \
   output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/

# 清理
rm output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/sol_ut.jsonl
rmdir $TEMP_DIR
```

**输出文件**：
```
output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/100_sol_100_ut_result.jsonl
```

**注意目录结构**：
- `llama3-8b_sol_coderm-8b_ut`：表示 solutions 来自 llama3-8b，unit tests 来自 coderm-8b

---

### 阶段 5: 计算 Table 2 指标

```bash
# 在本地（或服务器）
cd /Users/fyc/Desktop/CodeRM

# 计算指标（注意 ut_model 是 coderm-8b）
python evaluation/calculate_table2_metrics.py \
  --benchmark humaneval+ \
  --sol_model llama3-8b \
  --ut_model coderm-8b \
  --sol_num 100 \
  --ut_num 100 \
  --mode both \
  --output_dir output/table2_results
```

**输出文件**：
```
output/table2_results/humaneval+_llama3-8b_coderm-8b.json
```

**生成汇总**：
```bash
python evaluation/generate_table2_summary.py \
  --results_dir output/table2_results \
  --benchmark humaneval+ \
  --sol_model llama3-8b
```

---

## 📊 预期结果

根据论文 Table 2，CodeRM-8B 的预期结果：

### Individual Unit Tests
| 指标 | 预期值 |
|------|--------|
| Accuracy | 69.64 |
| F1 Score | 63.63 |
| FAR | 11.17 |
| FRR | 38.55 |

### Multiple Unit Tests
| 指标 | 预期值 |
|------|--------|
| Accuracy | 80.46 |
| F1 Score | 81.27 |
| FAR | 16.48 |
| FRR | 22.71 |

**注意**：CodeRM-8B 在 Multiple Unit Tests 模式下表现最好！

---

## 🔄 与 Llama3.1-8B 的对比

| 项目 | Llama3.1-8B | CodeRM-8B |
|------|-------------|------------|
| 配置文件 | `config_ut_llama3-8b.json` | `config_ut_coderm-8b.json` |
| 模型路径 | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `KAKA22/CodeRM-8B` |
| GPU 需求 | 1 张 GPU | 1 张 GPU |
| 输出目录 | `llama3-8b_unit_tests` | `coderm-8b_unit_tests` |
| Unit Test 文件 | `ut_llama3-8b_100.jsonl` | `ut_coderm-8b_100.jsonl` |
| Docker 输出目录 | `llama3-8b_sol_llama3-8b_ut` | `llama3-8b_sol_coderm-8b_ut` |
| 指标文件 | `humaneval+_llama3-8b_llama3-8b.json` | `humaneval+_llama3-8b_coderm-8b.json` |

---

## ✅ 检查清单

### 阶段 2 完成
- [ ] `output/inference/coderm-8b_unit_tests/output_gpu_0.jsonl` 存在
- [ ] `output/inference/coderm-8b_unit_tests/merge_result.jsonl` 存在
- [ ] `data/result/humaneval+/ut_coderm-8b_100.jsonl` 存在

### 阶段 3 完成
- [ ] 本地有 `data/result/humaneval+/ut_coderm-8b_100.jsonl`
- [ ] 本地有 `data/result/humaneval+/sol_llama3-8b_200_anno.jsonl`

### 阶段 4 完成
- [ ] `output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/100_sol_100_ut_result.jsonl` 存在
- [ ] 文件包含 1,640,000 行结果

### 阶段 5 完成
- [ ] `output/table2_results/humaneval+_llama3-8b_coderm-8b.json` 存在
- [ ] 指标值在合理范围内

---

## 🆘 常见问题

### 问题 1: CodeRM-8B 模型下载失败

**错误**：
```
404 Client Error: Not Found
```

**解决**：
1. 确认模型名称正确：`KAKA22/CodeRM-8B`
2. 检查 HuggingFace token 权限
3. 运行：`python test_token.py` 验证访问权限

### 问题 2: 目录名称不匹配

**错误**：
```
FileNotFoundError: .../llama3-8b_sol_coderm-8b_ut/...
```

**解决**：
- 确保目录名称正确：`llama3-8b_sol_coderm-8b_ut`
- 注意：solutions 来自 `llama3-8b`，unit tests 来自 `coderm-8b`

### 问题 3: 配置文件找不到

**错误**：
```
FileNotFoundError: config_ut_coderm-8b.json
```

**解决**：
- 确认文件存在：`ls inference/config_ut_coderm-8b.json`
- 如果不存在，检查文件名是否正确

---

## 🚀 快速命令参考

### 服务器端

```bash
# 生成 Unit Tests（CodeRM-8B）
cd inference
python inference_mp.py --config config_ut_coderm-8b.json

# 合并和提取
cd ../preprocess
python merge_output.py --mp_num 1 --input_dir ../output/inference/coderm-8b_unit_tests
python extract_unit_test.py --input_path ../output/inference/coderm-8b_unit_tests/merge_result.jsonl --id_path ../data/benchmark/input_humaneval+_ut.jsonl --output_path ../data/result/humaneval+/ut_coderm-8b_100.jsonl
```

### 本地

```bash
# Docker 执行（手动）
python3 -c "from evaluation.evaluate import save_sol_and_ut_comb; save_sol_and_ut_comb('humaneval+', 'llama3-8b', 'coderm-8b', 100, 100)"
TEMP_DIR=$(mktemp -d -t docker_write_XXXXXX)
chmod 777 "$TEMP_DIR"
docker run -v "$(pwd):/data" kaka0605/exec_unit_test:24.12.30 --input_path /data/output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/sol_ut.jsonl --output_path /data/$TEMP_DIR/100_sol_100_ut_result.jsonl --mp_num 8 --chunk_size 1000 --recover 0
mv $TEMP_DIR/100_sol_100_ut_result.jsonl output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/

# 计算指标
python evaluation/calculate_table2_metrics.py --benchmark humaneval+ --sol_model llama3-8b --ut_model coderm-8b --sol_num 100 --ut_num 100 --mode both
```

---

## 📝 总结

使用 CodeRM-8B 的流程与 Llama3.1-8B 基本相同，主要区别：

1. ✅ **配置文件**：使用 `config_ut_coderm-8b.json`
2. ✅ **模型路径**：`KAKA22/CodeRM-8B`
3. ✅ **输出目录**：`coderm-8b_unit_tests` 和 `llama3-8b_sol_coderm-8b_ut`
4. ✅ **指标计算**：`--ut_model coderm-8b`

**如果你已经完成了 Llama3.1-8B 的实验，只需要重新运行阶段 2-5 即可！**

祝实验顺利！🚀
