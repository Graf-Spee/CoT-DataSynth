#!/usr/bin/env bash

set_qwen_family_defaults() {
  MODEL_FAMILY="${MODEL_FAMILY:-qwen2_5}"

  case "${MODEL_FAMILY}" in
    qwen2_5)
      FAMILY_TAG="qwen2_5"
      DEFAULT_BASE_MODEL="/data/pretrain_models/Qwen2.5-3B-Instruct"
      DEFAULT_TEACHER_MODEL="/data/pretrain_models/Qwen2.5-7B-Instruct"
      ;;
    qwen3)
      FAMILY_TAG="qwen3"
      DEFAULT_BASE_MODEL="/data/pretrain_models/Qwen3-4B"
      DEFAULT_TEACHER_MODEL="/data/pretrain_models/Qwen3-8B"
      ;;
    qwen3_5)
      FAMILY_TAG="qwen3_5"
      DEFAULT_BASE_MODEL="/data/pretrain_models/Qwen3.5-4B"
      DEFAULT_TEACHER_MODEL="/data/pretrain_models/Qwen3.5-9B"
      ;;
    *)
      echo "[ERROR] Unknown MODEL_FAMILY=${MODEL_FAMILY}. Use qwen2_5, qwen3, or qwen3_5." >&2
      exit 1
      ;;
  esac

  BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
  if [ "${SET_TEACHER_MODEL:-1}" = "1" ]; then
    TEACHER_MODEL="${TEACHER_MODEL:-${DEFAULT_TEACHER_MODEL}}"
  fi

  export MODEL_FAMILY FAMILY_TAG BASE_MODEL
  if [ "${SET_TEACHER_MODEL:-1}" = "1" ]; then
    export TEACHER_MODEL
  fi
}
