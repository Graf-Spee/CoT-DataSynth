import json
import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm


def jsonl_to_parquet_with_val(
    input_path: str,
    train_output: str,
    val_output: str,
    val_ratio: float = 0.01,      # 验证集比例，默认 1%
    val_seed: int = 42,             # 随机种子（通过 hash 实现确定性抽样）
    batch_size: int = 5000,
):
    """
    流式读取 jsonl，按 val_ratio 随机分出验证集，分别保存为 parquet。
    每行格式: {"messages": [{"role":"user",...}, {"role":"assistant",...}]}
    输出列: question, answer（纯字符串，与训练兼容）
    """
    input_path = Path(input_path)
    train_output = Path(train_output)
    val_output = Path(val_output)

    schema = pa.schema([
        pa.field("question", pa.large_utf8()),
        pa.field("answer", pa.large_utf8())
    ])

    train_writer = pq.ParquetWriter(train_output, schema, compression='zstd')
    val_writer = pq.ParquetWriter(val_output, schema, compression='zstd')

    train_records = []
    val_records = []
    train_count = 0
    val_count = 0
    skipped = 0

    # 流式确定性随机：对每行内容做 hash，均匀映射到 [0, 1)
    def is_val(line_content: str) -> bool:
        h = hashlib.md5(f"{val_seed}:{line_content}".encode()).hexdigest()
        return int(h, 16) / (2**128) < val_ratio

    with open(input_path, 'r', encoding='utf-8') as f:
        # 估算总行数用于进度条（可选）
        for line in tqdm(f, desc="Processing", unit="lines"):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                messages = row.get("messages", [])
            except Exception:
                skipped += 1
                continue

            if len(messages) < 2:
                skipped += 1
                continue

            question = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            answer = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")

            record = {"question": question, "answer": answer}

            if is_val(line):
                val_records.append(record)
                val_count += 1
                if len(val_records) >= batch_size:
                    df_chunk = pd.DataFrame(val_records)
                    table = pa.Table.from_pandas(df_chunk, schema=schema)
                    val_writer.write_table(table)
                    val_records = []
            else:
                train_records.append(record)
                train_count += 1
                if len(train_records) >= batch_size:
                    df_chunk = pd.DataFrame(train_records)
                    table = pa.Table.from_pandas(df_chunk, schema=schema)
                    train_writer.write_table(table)
                    train_records = []

    # 写入最后一批
    if train_records:
        df_chunk = pd.DataFrame(train_records)
        table = pa.Table.from_pandas(df_chunk, schema=schema)
        train_writer.write_table(table)
    if val_records:
        df_chunk = pd.DataFrame(val_records)
        table = pa.Table.from_pandas(df_chunk, schema=schema)
        val_writer.write_table(table)

    train_writer.close()
    val_writer.close()

    print(f"\n✅ 处理完成")
    print(f"   训练集: {train_count} 条 → {train_output}")
    print(f"   验证集: {val_count} 条 → {val_output}")
    print(f"   跳过/异常: {skipped} 条")
    print(f"   验证比例: {val_count / (train_count + val_count) * 100:.2f}%")


if __name__ == "__main__":
    jsonl_to_parquet_with_val(
        input_path="/data/open_datasets/math-train-qwq-rs-n256/data.jsonl",
        train_output="/data/open_datasets/math-train-qwq-rs-n256/train.parquet",
        val_output="/data/open_datasets/math-train-qwq-rs-n256/val.parquet",
        val_ratio=0.01,      # 1% 作为验证集，约 1.1 万条
        val_seed=42,
        batch_size=5000,
    )