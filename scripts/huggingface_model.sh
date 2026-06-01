#!/bin/bash
# set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 <username/model_name> [model_name (for saving)]"
    exit 1
fi

REPO_NAME=$1

if [ $# -gt 1 ]; then
    MODEL_NAME=$2
else
    MODEL_NAME="${REPO_NAME##*/}"
fi

SAVE_PATH="/data/pretrain_models/$MODEL_NAME"

# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_DOWNLOAD_TIMEOUT=120
# export HF_HUB_MAX_WORKERS=1
# export HF_HUB_ENABLE_HF_TRANSFER=0

export HF_TOKEN= # export your HF token as environment var. to run this script

huggingface-cli download \
    $REPO_NAME \
    --local-dir $SAVE_PATH \
    --resume-download \

echo "模型已下载到: $SAVE_PATH"