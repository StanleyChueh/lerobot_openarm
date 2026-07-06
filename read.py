import pyarrow.parquet as pq
import pandas as pd

# ✅ Parquet 檔案路徑
parquet_path = '/home/csl/.cache/huggingface/lerobot/ethanCSL/0330ezgame/meta/tasks.parquet'

# ✅ 讀取 parquet 表格
table = pq.read_table(parquet_path)


# ✅ 顯示欄位名稱
print("📌 所有欄位名稱:")
for name in table.schema.names:
    print(f" - {name}")

# ✅ 轉為 pandas DataFrame
df = table.to_pandas()

# ✅ 預覽前幾筆資料
print("\n📊 資料預覽:")
print(df.head())


# ✅ 遍歷每列觀察資料
print("\n🔍 每列資料處理（示例）:")
for idx, row in df.iterrows():
    gripper_width = row.get('observation.gripper_width', None)
    observation_state = row.get('observation.state', None)
    action = row.get('action', None)
    print(f"[{idx}] gripper_width={gripper_width}, observation.state={observation_state}, action={action}")