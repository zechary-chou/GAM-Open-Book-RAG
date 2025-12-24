#!/bin/bash

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/bin/activate your_env
set -euo pipefail
source .venv/bin/activate

# Load .env if it exists
if [ -f .env ]; then
  set -a            # auto-export variables
  source .env
  set +a
else
  echo ".env file not found"
  exit 1
fi




# Set output directory
base_outputdir=./results/hotpotqa

# Create output directory
mkdir -p $base_outputdir

# Run HotpotQA evaluation
# for dataset in "eval_400" "eval_1600" "eval_3200"

for dataset in "eval_400"
do
    echo "Processing dataset: $dataset"
    outputdir=$base_outputdir/${dataset}

    # python3 eval/hotpotqa_test.py \
    #     --data ./data/hotpotqa/${dataset}.json \
    #     --outdir $outputdir \
    #     --start-idx 0 \
    #     --max-tokens 2048 \
    #     --memory-api-key "$OPENAI_API_KEY" \
    #     --memory-base-url "https://api.openai.com/v1" \
    #     --memory-model "gpt-4o-mini" \
    #     --memory-api-type "openai" \
    #     --research-api-key "$OPENAI_API_KEY" \
    #     --research-base-url "https://api.openai.com/v1" \
    #     --research-model "gpt-4o-mini" \
    #     --research-api-type "openai" \
    #     --working-api-key "$OPENAI_API_KEY" \
    #     --working-base-url "https://api.openai.com/v1" \
    #     --working-model "gpt-4o-mini" \
    #     --working-api-type "openai" \
    #     --embedding-model-path BAAI/bge-m3
    python3 eval/hotpotqa_test.py \
        --data ./data/hotpotqa/${dataset}.json \
        --outdir $outputdir \
        --start-idx 0 \
        --max-tokens 2048 \
        --memory-api-key "$OPENAI_API_KEY" \
        --memory-base-url "http://0.0.0.0:8000/v1" \
        --memory-model "Qwen/Qwen2.5-1.5B-Instruct" \
        --memory-api-type "vllm" \
        --research-api-key "$OPENAI_API_KEY" \
        --research-base-url "http://0.0.0.0:8000/v1" \
        --research-model "Qwen/Qwen2.5-1.5B-Instruct" \
        --research-api-type "vllm" \
        --working-api-key "$OPENAI_API_KEY" \
        --working-base-url "http://0.0.0.0:8000/v1" \
        --working-model "Qwen/Qwen2.5-1.5B-Instruct" \
        --working-api-type "vllm" \
        --embedding-model-path BAAI/bge-m3
done