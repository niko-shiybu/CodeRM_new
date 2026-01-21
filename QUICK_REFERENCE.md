# Llama3.1-8B 实验快速参考

## 🚀 快速命令清单

### 服务器端（GPU 推理）

```bash
# 1. 检查环境
bash check_gpu.sh
python test_token.py

# 2. 生成 Solutions
cd inference
python inference_mp.py --config config_sol_llama3-8b.json
cd ../preprocess
python merge_output.py --mp_num 1 --input_dir ../output/inference/llama3-8b_solutions
python extract_solution.py --data_path ../output/inference/llama3-8b_solutions/merge_result.jsonl --id_path ../data/benchmark/input_humaneval+_sol.jsonl --output_path ../data/result/humaneval+/sol_llama3-8b_200.jsonl

# 3. 生成 Unit Tests
cd ../inference
python inference_mp.py --config config_ut_llama3-8b.json
cd ../preprocess
python merge_output.py --mp_num 1 --input_dir ../output/inference/llama3-8b_unit_tests
python extract_unit_test.py --input_path ../output/inference/llama3-8b_unit_tests/merge_result.jsonl --id_path ../data/benchmark/input_humaneval+_ut.jsonl --output_path ../data/result/humaneval+/ut_llama3-8b_100.jsonl
```

### 数据传输（服务器 → 本地）

```bash
# 在本地 Mac 执行
scp user@server:/path/to/CodeRM/data/result/humaneval+/sol_llama3-8b_200_anno.jsonl ./data/result/humaneval+/
scp user@server:/path/to/CodeRM/data/result/humaneval+/ut_llama3-8b_100.jsonl ./data/result/humaneval+/
scp user@server:/path/to/CodeRM/data/benchmark/input_humaneval+_sol.jsonl ./data/benchmark/
scp user@server:/path/to/CodeRM/data/benchmark/input_humaneval+_ut.jsonl ./data/benchmark/
```

### 本地（Docker 执行）

```bash
# 一键运行
bash run_docker_local.sh

# 或手动执行
python3 -c "from evaluation.evaluate import save_sol_and_ut_comb; save_sol_and_ut_comb('humaneval+', 'llama3-8b', 'llama3-8b', 100, 100)"
TEMP_DIR=$(mktemp -d -t docker_write_XXXXXX)
chmod 777 "$TEMP_DIR"
docker run -v "$(pwd):/data" kaka0605/exec_unit_test:24.12.30 --input_path /data/output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/sol_ut.jsonl --output_path /data/$TEMP_DIR/100_sol_100_ut_result.jsonl --mp_num 8 --chunk_size 1000 --recover 0
mv $TEMP_DIR/100_sol_100_ut_result.jsonl output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/
```

### 计算指标

```bash
# 计算指标（快速）
python evaluation/calculate_table2_metrics.py --benchmark humaneval+ --sol_model llama3-8b --ut_model llama3-8b --sol_num 100 --ut_num 100 --mode both

# 生成汇总
python evaluation/generate_table2_summary.py --results_dir output/table2_results --benchmark humaneval+ --sol_model llama3-8b
```

## 📊 预期结果

| 模式 | Acc | F1 | FAR | FRR |
|------|-----|----|----|-----|
| Individual | 60.02 | 44.97 | 13.66 | 46.13 |
| Multiple | 74.21 | 74.35 | 20.44 | 30.55 |

## ⏱️ 时间估算

- GPU 推理：4-8 小时
- Docker 执行：2-6 小时
- 指标计算：10-30 分钟
- **总计**：6-15 小时

## 📁 关键文件路径

### 服务器端
- Solutions: `data/result/humaneval+/sol_llama3-8b_200_anno.jsonl`
- Unit Tests: `data/result/humaneval+/ut_llama3-8b_100.jsonl`

### 本地
- Docker 结果: `output/humaneval+/llama3-8b_sol_llama3-8b_ut/details/100_sol_100_ut_result.jsonl`
- 指标结果: `output/table2_results/humaneval+_llama3-8b_llama3-8b.json`

## 🔍 检查点

- [ ] GPU 可用：`nvidia-smi`
- [ ] Token 有效：`python test_token.py`
- [ ] Docker 运行：`docker ps`
- [ ] 数据文件存在：检查 `data/result/humaneval+/`

## 🆘 常见问题

- **GPU 检测失败**：降低阈值或手动指定 GPU
- **显存不足**：降低 `gpu_memory_utilization`
- **Docker 失败**：检查镜像和权限
- **文件缺失**：重新下载或生成

详细说明请查看：`COMPLETE_EXPERIMENT_GUIDE_LLAMA3_8B.md`
