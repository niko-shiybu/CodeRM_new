# Llama3.1-8B 完整实验流程（混合方案：服务器 GPU + 本地 Docker）

## 实验目标

复现 Table 2 中 **Llama3.1-8B** 作为 Reward Model 的实验结果：
- **Policy Model**: Llama3.1-8B（生成 solutions）
- **Reward Model**: Llama3.1-8B（生成 unit tests）
- **Benchmark**: HumanEval+
- **指标**: Accuracy, F1, FAR, FRR, Line Coverage

## 前置条件

### 服务器端
- ✅ GPU 可用（至少 1 张，16GB+ 显存）
- ✅ Conda 环境已配置
- ✅ HuggingFace token 已配置（已验证）

### 本地（Mac）
- ✅ Docker Desktop 已安装并运行
- ✅ Python 环境（用于预处理和指标计算）

## 完整流程

### 阶段 1: 服务器 - 生成 Solutions（GPU）

```bash
# 在服务器上
cd /path/to/CodeRM
conda activate coderm

# 1. 生成 solutions
cd inference
python inference_mp.py --config config_sol_llama3-8b.json

# 2. 合并输出（根据实际使用的 GPU 数量调整 mp_num）
cd ../preprocess
python merge_output.py --mp_num 1 --input_dir ../output/inference/llama3-8b_solutions

# 3. 提取 solutions
python extract_solution.py \
  --data_path ../output/inference/llama3-8b_solutions/merge_result.jsonl \
  --id_path ../data/benchmark/input_humaneval+_sol.jsonl \
  --output_path ../data/result/humaneval+/sol_llama3-8b_200.jsonl
```

**注意**：如果你已经有 `sol_llama3-8b_200_anno.jsonl`（标注文件），可以跳过生成步骤，直接使用现有文件。

### 阶段 2: 服务器 - 生成 Unit Tests（GPU）

```bash
# 在服务器上
cd inference
python inference_mp.py --config config_ut_llama3-8b.json

# 合并和提取
cd ../preprocess
python merge_output.py --mp_num 1 --input_dir ../output/inference/llama3-8b_unit_tests
python extract_unit_test.py \
  --input_path ../output/inference/llama3-8b_unit_tests/merge_result.jsonl \
  --id_path ../data/benchmark/input_humaneval+_ut.jsonl \
  --output_path ../data/result/humaneval+/ut_llama3-8b_100.jsonl
```

### 阶段 3: 传输数据到本地

```bash
# 在本地（Mac）执行
# 替换 user@server:/path/to/CodeRM 为你的实际服务器地址

# 1. 下载 solutions（如果需要）
scp user@server:/path/to/CodeRM/data/result/humaneval+/sol_llama3-8b_200_anno.jsonl \
   ./data/result/humaneval+/

# 2. 下载 unit tests
scp user@server:/path/to/CodeRM/data/result/humaneval+/ut_llama3-8b_100.jsonl \
   ./data/result/humaneval+/

# 3. 确保 benchmark 数据存在（如果还没有）
mkdir -p ./data/benchmark
scp user@server:/path/to/CodeRM/data/benchmark/input_humaneval+_sol.jsonl \
   ./data/benchmark/
scp user@server:/path/to/CodeRM/data/benchmark/input_humaneval+_ut.jsonl \
   ./data/benchmark/
```

### 阶段 4: 本地 - 执行 Docker（执行单元测试）

#### 方法 A: 使用自动化脚本（推荐）

```bash
# 在本地
cd /Users/fyc/Desktop/CodeRM

# 运行脚本
bash run_docker_local.sh
```

#### 方法 B: 手动执行

```bash
# 在本地
cd /Users/fyc/Desktop/CodeRM

# 1. 拉取 Docker 镜像（如果还没有）
docker pull kaka0605/exec_unit_test:24.12.30

# 2. 创建输出目录
mkdir -p output/humaneval+/llama3-8b_sol_llama3-8b_ut/details

# 3. 生成 sol_ut.jsonl
python3 << 'PYTHON_SCRIPT'
from evaluation.evaluate import save_sol_and_ut_comb
save_sol_and_ut_comb('humaneval+', 'llama3-8b', 'llama3-8b', 100, 100)
PYTHON_SCRIPT

# 4. 执行 Docker
TEMP_DIR=$(mktemp -d -t docker_write_XXXXXX)
chmod 777 "$TEMP_DIR"

docker run -v "$(pwd):/data" kaka0605/exec_unit_test:24.12.30 \
    --input_path /data/output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/sol_ut.jsonl \
    --output_path /data/$TEMP_DIR/100_sol_100_ut_result.jsonl \
    --mp_num 8 \
    --chunk_size 1000 \
    --recover 0

# 5. 移动结果
mv $TEMP_DIR/100_sol_100_ut_result.jsonl \
   output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/

# 6. 清理
rm output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/sol_ut.jsonl
rmdir $TEMP_DIR
```

**预计时间**：Docker 执行可能需要几小时（取决于你的 Mac 性能）

### 阶段 5: 计算 Table 2 指标

#### 选项 A: 在本地计算（推荐）

```bash
# 在本地
cd /Users/fyc/Desktop/CodeRM

# 计算指标（不计算覆盖率，更快）
python evaluation/calculate_table2_metrics.py \
  --benchmark humaneval+ \
  --sol_model llama3-8b \
  --ut_model llama3-8b \
  --sol_num 100 \
  --ut_num 100 \
  --mode both \
  --output_dir output/table2_results

# 生成汇总表格
python evaluation/generate_table2_summary.py \
  --results_dir output/table2_results \
  --benchmark humaneval+ \
  --sol_model llama3-8b
```

#### 选项 B: 传输到服务器计算

```bash
# 1. 传输结果文件到服务器
scp output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/100_sol_100_ut_result.jsonl \
   user@server:/path/to/CodeRM/output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/

# 2. 在服务器上计算指标
# （在服务器上执行）
python evaluation/calculate_table2_metrics.py \
  --benchmark humaneval+ \
  --sol_model llama3-8b \
  --ut_model llama3-8b \
  --sol_num 100 \
  --ut_num 100 \
  --mode both
```

## 预期结果

根据论文 Table 2，Llama3.1-8B 的预期结果：

### Individual Unit Tests
- Accuracy: 60.02
- F1: 44.97
- FAR: 13.66
- FRR: 46.13

### Multiple Unit Tests
- Accuracy: 74.21
- F1: 74.35
- FAR: 20.44
- FRR: 30.55

## 时间估算

- **服务器 GPU 推理**（生成 solutions + unit tests）：约 2-4 小时
- **本地 Docker 执行**（执行单元测试）：约 2-6 小时（取决于 Mac 性能）
- **指标计算**：约 10-30 分钟

**总计**：约 4-10 小时

## 故障排查

### 问题 1: Docker 执行失败

**检查**：
```bash
docker ps
docker images | grep exec_unit_test
```

**解决**：
- 确保 Docker Desktop 正在运行
- 重新拉取镜像：`docker pull kaka0605/exec_unit_test:24.12.30`

### 问题 2: 数据文件缺失

**检查**：
```bash
ls -lh data/result/humaneval+/sol_llama3-8b_200_anno.jsonl
ls -lh data/result/humaneval+/ut_llama3-8b_100.jsonl
```

**解决**：从服务器重新下载文件

### 问题 3: 路径不一致

**解决**：确保本地和服务器使用相同的相对路径结构

## 下一步

完成 Llama3.1-8B 后，可以继续：
1. **Llama3.1-70B**：需要 4 张 GPU，重复相同流程
2. **CodeRM-8B**：需要 1 张 GPU，重复相同流程

然后运行 `generate_table2_summary.py` 生成完整的 Table 2。
