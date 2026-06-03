# Repo for Synthetic CoT Data

All scripts for training and evaluation are in `/scripts`. Please modify `config/bash_config.env` to set the directory for storing model checkpoints and logs. Usage of the scripts are instructed upon running `bash <script_name.sh>` without any parameters.


## `scripts/cot_distill_teacher_filter.py`

This script generates CoT responses with a local teacher model through vLLM, then filters generated samples by checking whether the teacher answer matches the gold answer. It currently supports `commonsenseqa` and `mbpp`.

Input parquet requirements:

- `prompt`: the original question or prompt.
- `reward_model`: a dict-like value containing `ground_truth`.

Basic usage:

```bash
python scripts/cot_distill_teacher_filter.py \
  --dataset commonsenseqa \
  --input-file /path/to/input.parquet \
  --output-file /path/to/output.parquet \
  --model-id /path/to/teacher-model \
  --gpu-ids 0 \
  --num-cots 4 \
  --do-sample
```

Common parameters:

- `--dataset`: dataset name, either `commonsenseqa` or `mbpp`.
- `--input-file`: processed input parquet file.
- `--output-file`: output parquet file for filtered CoT samples.
- `--model-id`: local teacher model path or Hugging Face model id.
- `--gpu-ids`: visible GPU ids, for example `0` or `0,1`.
- `--num-cots`: number of CoT candidates generated for each input row. Values greater than `1` require `--do-sample`.
- `--gen-batch-size`: vLLM generation batch size. Lower it if GPU memory is insufficient.
- `--tensor-parallel-size`: vLLM tensor parallel size, usually matching the number of GPUs used by the model.
- `--smoke-num-rows`: process only the first N rows for a quick test.
- `--disable-teacher-filter`: keep all generated CoTs and add `teacher_filter_passed` to indicate whether each sample passed the filter.

The output parquet contains `question`, `answer`, and `gold_answer`. When `--disable-teacher-filter` is set, it also contains `teacher_filter_passed`.
