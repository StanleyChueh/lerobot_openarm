# # import pandas as pd

# # # 1. Your exact file path
# # parquet_path = '/home/csl/.cache/huggingface/lerobot/ethanCSL/0328result/meta/tasks.parquet'

# # # 2. Read the dataframe
# # df = pd.read_parquet(parquet_path)

# # # 3. Handle the Index (Based on your printout, 'task' might be acting as the index)
# # if 'task' not in df.columns:
# #     df = df.reset_index()

# # # 4. Perform the update
# # old_task = 'tttttt'
# # new_task = 'pick up the left cup place it on the right one'

# # # Check if we found it
# # matches = (df['task'] == old_task).sum()
# # print(f"Found {matches} row(s) matching '{old_task}'.")

# # # Replace it
# # df.loc[df['task'] == old_task, 'task'] = new_task

# # # 5. Overwrite the file (index=False ensures we keep the Hugging Face format clean)
# # df.to_parquet(parquet_path, index=False)

# # print("✅ Parquet file updated successfully!")



# from huggingface_hub import HfApi

# # Initialize the API
# api = HfApi()

# # 1. Your local file path
# local_file_path = '/home/csl/.cache/huggingface/lerobot/ethanCSL/0328result/meta/tasks.parquet'

# # 2. Your Hugging Face repository ID (Username / Dataset Name)
# repo_id = "ethanCSL/0328result" 

# # 3. The path where this file should be saved INSIDE the repo
# path_in_repo = "meta/tasks.parquet"

# print(f"Uploading {local_file_path} to {repo_id}...")

# # 4. Push to the hub!
# api.upload_file(
#     path_or_fileobj=local_file_path,
#     path_in_repo=path_in_repo,
#     repo_id=repo_id,
#     repo_type="dataset" # Crucial: tells HF this is a dataset, not a model
# )

# print("✅ Successfully pushed to the Hub!")
import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "ethanCSL/0328result"
filename = "meta/tasks.parquet"

print(f"Fetching {filename} directly from {repo_id}...")

try:
    file_path = hf_hub_download(
        repo_id=repo_id, 
        repo_type="dataset", 
        filename=filename,
        force_download=True 
    )
    
    df = pd.read_parquet(file_path)
    print("\n📊 --- Data currently on the Hub ---")
    print(df)
    print("---------------------------------\n")

    if 'task' not in df.columns:
         print("❌ ERROR: The 'task' column is missing entirely!")
    elif df['task'].isnull().any():
        print("❌ ERROR: There are 'None' or 'NaN' values in your task column!")
    else:
        print("✅ The 'task' column exists and has no None values.")

except Exception as e:
    print(f"❌ Failed to download or read the file: {e}")