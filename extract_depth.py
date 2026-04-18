import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from tqdm import tqdm

# ==========================================
# 1. 設定對應您現有結構的路徑
# ==========================================
# 這是您現在彩色圖片的所在地
INPUT_DIR = "dataset/scenery/train/images"

# 這是我們要新建來放深度圖的所在地
OUTPUT_DIR = "dataset/scenery/train/depth_maps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 載入 Depth Anything V2 模型
# ==========================================
print("載入 Depth Anything V2 模型中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=device
)

# ==========================================
# 3. 開始批次處理
# ==========================================
image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"找到 {len(image_files)} 張圖片，開始萃取深度圖...")

for filename in tqdm(image_files):
    img_path = os.path.join(INPUT_DIR, filename)

    try:
        image = Image.open(img_path).convert('RGB')
        depth_result = depth_estimator(image)
        depth_image = depth_result["depth"]

        depth_array = np.array(depth_image)
        depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 強制存成 .png (保留最高品質深度資訊)
        out_name = filename.rsplit('.', 1)[0] + '.png'
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, depth_normalized)

    except Exception as e:
        print(f"處理 {filename} 時發生錯誤: {e}")

print("🎉 訓練集深度圖萃取完成！")