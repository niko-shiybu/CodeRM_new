#!/bin/bash

# Script to reproduce Table 2 experiments
# This script runs the complete pipeline for Table 2: Quality of Unit Tests

set -e  # Exit on error

# Configuration
BENCHMARK="humaneval+"
SOL_MODEL="llama3-8b"
UT_MODELS=("llama3-8b" "llama3-70b" "coderm-8b")
SOL_NUM=100
UT_NUM=100
MP_NUM=8  # Number of processes for code execution

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Table 2 Experiment: Quality of Unit Tests${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Benchmark: $BENCHMARK"
echo "Policy Model (Solution Generator): $SOL_MODEL"
echo "Reward Models (Unit Test Generators): ${UT_MODELS[@]}"
echo ""

# Step 1: Check if solutions already exist
echo -e "${YELLOW}Step 1: Checking for existing solutions...${NC}"
SOL_ANNO_FILE="data/result/${BENCHMARK}/sol_${SOL_MODEL}_200_anno.jsonl"
if [ ! -f "$SOL_ANNO_FILE" ]; then
    echo -e "${RED}Warning: Solution annotations not found at $SOL_ANNO_FILE${NC}"
    echo "You need to generate solutions first using inference_mp.py"
    echo "Example:"
    echo "  cd inference"
    echo "  python inference_mp.py --config config_sol_llama3-8b.json"
    echo "  cd ../preprocess"
    echo "  python merge_output.py --mp_num <num_processes> --input_dir ../output/inference/llama3-8b_solutions"
    echo "  python extract_solution.py --data_path ../output/inference/llama3-8b_solutions/merge_result.jsonl --id_path ../data/benchmark/input_${BENCHMARK}_sol.jsonl --output_path ../data/result/${BENCHMARK}/sol_${SOL_MODEL}_200.jsonl"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
else
    echo -e "${GREEN}✓ Solutions found${NC}"
fi

# Step 2: Generate unit tests for each reward model
echo ""
echo -e "${YELLOW}Step 2: Generating unit tests...${NC}"
for UT_MODEL in "${UT_MODELS[@]}"; do
    echo ""
    echo -e "${YELLOW}Generating unit tests with ${UT_MODEL}...${NC}"
    
    UT_FILE="data/result/${BENCHMARK}/ut_${UT_MODEL}_100.jsonl"
    
    if [ -f "$UT_FILE" ]; then
        echo -e "${GREEN}✓ Unit tests already exist for ${UT_MODEL}${NC}"
    else
        echo "Generating unit tests..."
        cd inference
        CONFIG_FILE="config_ut_${UT_MODEL}.json"
        if [ ! -f "$CONFIG_FILE" ]; then
            echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
            exit 1
        fi
        
        echo "Running inference with config: $CONFIG_FILE"
        python inference_mp.py --config "$CONFIG_FILE"
        
        # Merge outputs
        cd ../preprocess
        OUTPUT_DIR="../output/inference/${UT_MODEL}_unit_tests"
        if [ -d "$OUTPUT_DIR" ]; then
            # Count GPU output files
            MP_COUNT=$(ls -1 ${OUTPUT_DIR}/output_gpu_*.jsonl 2>/dev/null | wc -l)
            if [ "$MP_COUNT" -gt 0 ]; then
                python merge_output.py --mp_num "$MP_COUNT" --input_dir "$OUTPUT_DIR"
                
                # Extract unit tests
                python extract_unit_test.py \
                    --input_path "${OUTPUT_DIR}/merge_result.jsonl" \
                    --id_path "../data/benchmark/input_${BENCHMARK}_ut.jsonl" \
                    --output_path "../data/result/${BENCHMARK}/ut_${UT_MODEL}_100.jsonl"
                
                echo -e "${GREEN}✓ Unit tests generated for ${UT_MODEL}${NC}"
            else
                echo -e "${RED}Error: No output files found in $OUTPUT_DIR${NC}"
            fi
        else
            echo -e "${RED}Error: Output directory not found: $OUTPUT_DIR${NC}"
        fi
        cd ..
    fi
done

# Step 3: Execute unit tests
echo ""
echo -e "${YELLOW}Step 3: Executing unit tests...${NC}"
for UT_MODEL in "${UT_MODELS[@]}"; do
    echo ""
    echo -e "${YELLOW}Executing unit tests for ${UT_MODEL}...${NC}"
    
    RESULT_DIR="output/${BENCHMARK}/${SOL_MODEL}_sol_${UT_MODEL}_ut/details"
    RESULT_FILE="${RESULT_DIR}/${SOL_NUM}_sol_${UT_NUM}_ut_result.jsonl"
    
    if [ -f "$RESULT_FILE" ]; then
        echo -e "${GREEN}✓ Execution results already exist for ${UT_MODEL}${NC}"
    else
        echo "Executing unit tests..."
        python evaluation/evaluate.py \
            --benchmark "$BENCHMARK" \
            --sol_model "$SOL_MODEL" \
            --ut_model "$UT_MODEL" \
            --sol_num "$SOL_NUM" \
            --ut_num "$UT_NUM" \
            --mp_num "$MP_NUM"
        
        if [ -f "$RESULT_FILE" ]; then
            echo -e "${GREEN}✓ Execution completed for ${UT_MODEL}${NC}"
        else
            echo -e "${RED}Error: Execution results not found${NC}"
        fi
    fi
done

# Step 4: Calculate Table 2 metrics
echo ""
echo -e "${YELLOW}Step 4: Calculating Table 2 metrics...${NC}"
echo ""

RESULTS_DIR="output/table2_results"
mkdir -p "$RESULTS_DIR"

for UT_MODEL in "${UT_MODELS[@]}"; do
    echo ""
    echo -e "${YELLOW}Calculating metrics for ${UT_MODEL}...${NC}"
    
    python evaluation/calculate_table2_metrics.py \
        --benchmark "$BENCHMARK" \
        --sol_model "$SOL_MODEL" \
        --ut_model "$UT_MODEL" \
        --sol_num "$SOL_NUM" \
        --ut_num "$UT_NUM" \
        --mode both \
        --output_dir "$RESULTS_DIR"
done

# Step 5: Generate summary table
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Experiment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "To view results, check the JSON files in $RESULTS_DIR"
echo "Or run:"
echo "  python evaluation/generate_table2_summary.py --results_dir $RESULTS_DIR"
