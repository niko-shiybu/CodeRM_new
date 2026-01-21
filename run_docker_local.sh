#!/bin/bash

# 脚本：在本地运行 Docker 执行单元测试
# 用途：当服务器没有 Docker 权限时，在本地执行 Docker 部分

set -e

# 配置
BENCHMARK="humaneval+"
SOL_MODEL="llama3-8b"
UT_MODEL="llama3-8b"
SOL_NUM=100
UT_NUM=100
MP_NUM=8  # Docker 并行进程数

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}本地 Docker 执行单元测试${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Benchmark: $BENCHMARK"
echo "Solution Model: $SOL_MODEL"
echo "Unit Test Model: $UT_MODEL"
echo "Solutions per task: $SOL_NUM"
echo "Unit tests per task: $UT_NUM"
echo ""

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    echo "请先安装 Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker ps &> /dev/null; then
    echo -e "${RED}错误: Docker 未运行或无权限${NC}"
    echo "请启动 Docker Desktop 或检查权限"
    exit 1
fi

echo -e "${GREEN}✓ Docker 可用${NC}"
echo ""

# 检查 Docker 镜像
echo -e "${YELLOW}检查 Docker 镜像...${NC}"
if ! docker images | grep -q "kaka0605/exec_unit_test.*24.12.30"; then
    echo "镜像不存在，正在拉取..."
    docker pull kaka0605/exec_unit_test:24.12.30
else
    echo -e "${GREEN}✓ 镜像已存在${NC}"
fi
echo ""

# 检查必要的数据文件
echo -e "${YELLOW}检查数据文件...${NC}"

SOL_ANNO_FILE="data/result/${BENCHMARK}/sol_${SOL_MODEL}_200_anno.jsonl"
UT_FILE="data/result/${BENCHMARK}/ut_${UT_MODEL}_100.jsonl"

if [ ! -f "$SOL_ANNO_FILE" ]; then
    echo -e "${RED}错误: 解决方案文件不存在: $SOL_ANNO_FILE${NC}"
    echo "请先从服务器下载此文件"
    exit 1
fi

if [ ! -f "$UT_FILE" ]; then
    echo -e "${RED}错误: 单元测试文件不存在: $UT_FILE${NC}"
    echo "请先从服务器下载此文件"
    exit 1
fi

echo -e "${GREEN}✓ 数据文件存在${NC}"
echo ""

# 创建输出目录
OUTPUT_DIR="output/${BENCHMARK}/${SOL_MODEL}_sol_${UT_MODEL}_ut/details"
mkdir -p "$OUTPUT_DIR"

# 步骤 1: 生成 sol_ut.jsonl
echo -e "${YELLOW}步骤 1: 生成解决方案-单元测试组合文件...${NC}"
python3 << PYTHON_SCRIPT
import sys
sys.path.insert(0, '.')
from evaluation.evaluate import save_sol_and_ut_comb

save_sol_and_ut_comb('${BENCHMARK}', '${SOL_MODEL}', '${UT_MODEL}', ${SOL_NUM}, ${UT_NUM})
print("✓ sol_ut.jsonl 生成完成")
PYTHON_SCRIPT

if [ ! -f "$OUTPUT_DIR/sol_ut.jsonl" ]; then
    echo -e "${RED}错误: sol_ut.jsonl 生成失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓ sol_ut.jsonl 已生成${NC}"
echo ""

# 步骤 2: 创建临时目录用于 Docker 输出
echo -e "${YELLOW}步骤 2: 准备 Docker 执行环境...${NC}"
TEMP_DIR=$(mktemp -d -t docker_write_XXXXXX)
chmod 777 "$TEMP_DIR"
echo "临时目录: $TEMP_DIR"
echo ""

# 步骤 3: 执行 Docker 容器
echo -e "${YELLOW}步骤 3: 执行单元测试（这可能需要较长时间）...${NC}"
echo "开始时间: $(date)"

docker run -v "$(pwd):/data" kaka0605/exec_unit_test:24.12.30 \
    --input_path "/data/$OUTPUT_DIR/sol_ut.jsonl" \
    --output_path "/data/$TEMP_DIR/${SOL_NUM}_sol_${UT_NUM}_ut_result.jsonl" \
    --mp_num $MP_NUM \
    --chunk_size 1000 \
    --recover 0

echo "结束时间: $(date)"
echo ""

# 步骤 4: 移动结果文件
echo -e "${YELLOW}步骤 4: 保存结果...${NC}"
if [ -f "$TEMP_DIR/${SOL_NUM}_sol_${UT_NUM}_ut_result.jsonl" ]; then
    mv "$TEMP_DIR/${SOL_NUM}_sol_${UT_NUM}_ut_result.jsonl" "$OUTPUT_DIR/"
    echo -e "${GREEN}✓ 结果文件已保存${NC}"
else
    echo -e "${RED}错误: 结果文件未生成${NC}"
    exit 1
fi

# 步骤 5: 清理
echo -e "${YELLOW}步骤 5: 清理临时文件...${NC}"
rm -f "$OUTPUT_DIR/sol_ut.jsonl"
rmdir "$TEMP_DIR" 2>/dev/null || true
echo -e "${GREEN}✓ 清理完成${NC}"
echo ""

# 完成
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Docker 执行完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "结果文件: $OUTPUT_DIR/${SOL_NUM}_sol_${UT_NUM}_ut_result.jsonl"
echo ""
echo "下一步："
echo "  1. 如果要在本地计算指标："
echo "     python evaluation/calculate_table2_metrics.py \\"
echo "       --benchmark $BENCHMARK \\"
echo "       --sol_model $SOL_MODEL \\"
echo "       --ut_model $UT_MODEL \\"
echo "       --sol_num $SOL_NUM \\"
echo "       --ut_num $UT_NUM \\"
echo "       --mode both"
echo ""
echo "  2. 如果要传输到服务器计算："
echo "     scp $OUTPUT_DIR/${SOL_NUM}_sol_${UT_NUM}_ut_result.jsonl \\"
echo "         user@server:/path/to/CodeRM/$OUTPUT_DIR/"
