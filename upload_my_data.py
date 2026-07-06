import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# 1. 設定你的路徑與帳號
# 請確保這個路徑下有 'data' 資料夾、'meta' 資料夾等
local_dir = "/home/csl/.cache/huggingface/lerobot/ethanCSL/0409tist"
repo_id = "ethanCSL/0409tist"

def upload():
    print(f"正在從 {local_dir} 載入資料集...")
    
    # 載入本地資料集物件
    # root 指向包含 metadata 和 parquet 檔案的目錄
    dataset = LeRobotDataset(repo_id, root=local_dir)
    
    # 執行上傳
    print(f"正在上傳至 Hugging Face: {repo_id}")
    dataset.push_to_hub()
    print("上傳成功！")

if __name__ == "__main__":
    upload()