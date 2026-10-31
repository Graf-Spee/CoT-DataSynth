import pandas as pd
from transformers import AutoTokenizer

model_path = "/home/hrh/data/pretrain_models/Qwen3-4B"
data_path = "/home/hrh/data/COT/experiments/exp9100_math_qwen3_4b_teacher14b_n4_sft_rl_for_aime25/distill/filtered_sft.parquet"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
VOCAB_SIZE = tokenizer.vocab_size
print(f"Qwen3 vocab size: {VOCAB_SIZE}")

df = pd.read_parquet(data_path)
total = len(df)
print(f"Total samples: {total}")

bad_count = 0
bad_index_list = []

for idx, row in df.iterrows():
    q = row["question"]
    a = row["answer"]
    full_text = tokenizer.apply_chat_template([
        {"role":"user","content":q},
        {"role":"assistant","content":a}
    ], tokenize=False, add_generation_prompt=False)
    ids = tokenizer(full_text)["input_ids"]

    max_id = max(ids)
    if max_id >= VOCAB_SIZE:
        bad_count += 1
        bad_index_list.append(idx)

    if len(ids) > 4096:
        print(f"[WARN] idx={idx} seq_len={len(ids)} >4096")

print("\n===== 统计结果 =====")
print(f"越界样本总数：{bad_count}")
print(f"总样本数：{total}")
if total > 0:
    print(f"越界占比：{bad_count / total:.2%}")

if bad_count > 0:
    print(f"越界样本索引前20个：{bad_index_list[:20]}")
else:
    print("✅ 没有发现token越界样本！")
